from torch import nn
from mmfewshot.detection.models.utils.aggregation_layer import AGGREGATORS
from mmcv.runner import BaseModule
from .apa import APA

import math
import torch
import torch.nn.functional as F
import sys
import os

from mmfewshot.detection.datasets.coco import COCO_SPLIT
from mmfewshot.detection.datasets.voc import VOC_SPLIT
import clip

def get_current_config():
    """Get the current running config file path"""
    # First try to get from environment variable
    config_path = os.environ.get('SVFA_CONFIG_PATH')
    if config_path:
        return config_path
        
    # If not in environment variable, get from command line arguments
    for arg in sys.argv:
        # Skip the first argument (script name)
        if arg.endswith('.py') and arg == sys.argv[0]:
            continue
        
        # If argument is a .py config file
        if arg.endswith('.py') and ('configs/' in arg or 'config/' in arg):
            return arg
        
        # If argument contains path separator and file extension, might be config file
        if '/' in arg and '.' in arg.split('/')[-1] and ('configs/' in arg or 'config/' in arg):
            return arg
    
    # If config file not found, print warning
    print("Warning: Could not find config file path in command line arguments")
    return None

with torch.no_grad():
    # Load CLIP model to CPU first
    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    
    def get_novel_classes(config):
        # Handle case when config is None
        if config is None:
            print("Warning: Config file path is None, cannot determine novel_classes")
            return None
            
        # If it's a temporary wrapper config file, try to read its content to get original config file path
        if 'tmp_wrapper_' in config:
            try:
                with open(config, 'r') as f:
                    content = f.readlines()
                    for line in content:
                        if "_base_ = ['" in line or '_base_ = ["' in line:
                            # Extract original config file path
                            original_config = line.split("_base_ = [")[1].strip().strip("'").strip('"').strip("]").strip("'").strip('"')
                            # Convert relative path to absolute path if needed
                            if not os.path.isabs(original_config):
                                original_config = os.path.join(os.path.dirname(config), original_config)
                            print(f"Found original config file: {original_config}")
                            config = original_config
                            break
            except Exception as e:
                print(f"Warning: Failed to read temporary wrapper config file: {e}")
            
        # Define mapping from config file to NOVEL_CLASSES
        voc_split_mapping = {
            "split1": VOC_SPLIT['NOVEL_CLASSES_SPLIT1'],
            "split2": VOC_SPLIT['NOVEL_CLASSES_SPLIT2'],
            "split3": VOC_SPLIT['NOVEL_CLASSES_SPLIT3'],
        }
        
        # Check VOC config files
        for split, novel_classes in voc_split_mapping.items():
            if f"voc-{split}_" in config:
                print(f"Found VOC split{split} novel_classes")
                return novel_classes
        
        # Check COCO config files
        if "coco" in config:
            print("Found COCO novel_classes")
            return COCO_SPLIT['NOVEL_CLASSES']
        
        print(f"Warning: Cannot determine novel_classes from config file {config}")
        return None

    # Semantic replacement list
    replacements = [
        "a photo of a",
        "an image of a",
        "a picture of a",
        "a snapshot of a",
        "a visual representation of a"
    ]

    # Define function to get CLIP features lazily
    def get_clip_features():
        config_path = get_current_config()
        novel_classes = get_novel_classes(config_path) if config_path else None
        
        # If no novel_classes found, return empty list
        if not novel_classes:
            print("Warning: No novel_classes found, returning empty clip_features")
            return []
            
        # Generate CLIP text feature vectors for each novel class name
        clip_features = []
        
        # Get current device - ensure we get it in get_clip_features, not globally
        current_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for class_name in novel_classes:
            # Create a list to store 5 text description features
            class_features = []
            
            for replacement in replacements:
                text = clip.tokenize([f"{replacement} {class_name}"]).to(current_device)  # Generate new text description
                # Temporarily move model to device for encoding, then move back to CPU
                clip_model.to(current_device)
                text_features = clip_model.encode_text(text)  # Get text features
                clip_model.to("cpu")  # Move back to CPU after encoding
                
                class_features.append(text_features.cpu())  # Move features back to CPU
            
            # Concatenate the 5 features of this class into a tensor
            class_features_tensor = torch.cat(class_features, dim=0)  # Shape: (5, 512)
            clip_features.append(class_features_tensor)
            
        # Check if there are features
        if clip_features:
            # Concatenate all class features into a tensor with shape (num_novel, 5, feature_dim)
            clip_features = torch.stack(clip_features, dim=0)  # (num_novel, 5, 512)
            # Move to device as needed
            return clip_features
        else:
            return []

@AGGREGATORS.register_module()
class PrototypesDistillation(BaseModule):
    def __init__(self, num_queries, dim, num_base_cls=15, num_novel=0):
        super().__init__()
        self.num_queries = num_queries
        self.num_base_cls = num_base_cls
        self.num_novel = num_novel
        self.dim = dim

        k_dim = dim // 4
        self.k_dim = k_dim

        self.query_embed = nn.Embedding(num_queries * num_base_cls, k_dim)
        self.duplicated = False
        if self.num_novel > 0:
            self.query_embed_novel = nn.Embedding(num_queries * num_novel, k_dim)
        self.w_qk = nn.Linear(dim, k_dim, bias=False)
        self.clip_proj = nn.Linear(512, 256)  # Let the model manage device automatically
        
    def forward(self, support_feats, support_gt_labels=None, forward_novel=False, forward_novel_test=False):
        """
        Args:
            support_feats: Tensor with shape (B, C, H, W).
            support_gt_labels: Support gt labels.
            forward_novel (bool): Novel classes.
            forward_novel_test (bool): Test time.
        Returns:
            tensor with shape (15, 1024)
        """

        # at the fine-tuning stage, duplicate the most compatible feature queries for the novel classes
        # ************************************************************
        if not self.duplicated and self.num_novel > 0 and not forward_novel_test and forward_novel:
            with torch.no_grad():
                support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
                B, C, H, W = support_feats_mp.shape
                k = support_feats_mp.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)
                k = self.w_qk(k)  # (B, 196, 1024)
                query_emb = self.query_embed.weight
                q = query_emb.unsqueeze(0).repeat(B, 1, 1)

                B, Nt, E = q.shape
                attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
                weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)

                drop = 5
                top_indices = torch.topk(weight, self.num_queries + drop, dim=-1)[1][:, -self.num_queries:]
                top_emb = torch.gather(self.query_embed.weight.unsqueeze(0).expand(B, -1, -1), 1, top_indices.unsqueeze(-1).expand(-1, -1, self.k_dim))
                
                # Get clip features
                clip_features = get_clip_features()
                
                # If clip features are found, apply them
                if len(clip_features) > 0:
                    # Add CLIP text features to query vectors
                    new_clip_features = clip_features
                
                    # Convert to Float type
                    new_clip_features = new_clip_features.to(torch.float32)  # Force convert to Float32
                    
                    # Move to the same device as top_emb
                    new_clip_features = new_clip_features.to(top_emb.device)

                    # Apply clip_proj to reduce feature dimension from 512 to 256
                    new_clip_features = self.clip_proj(new_clip_features)  # (B, num_queries, 256)
                    
                    top_emb = top_emb * new_clip_features
                else:
                    print("Warning: Cannot get clip_features, using original top_emb")
                
                top_emb = top_emb[torch.sort(support_gt_labels, dim=0)[1]].reshape(self.num_novel * self.num_queries, self.k_dim)
                self.query_embed_novel.weight.copy_(top_emb)
            self.duplicated = True  # set as True once duplicated

        support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
        B, C, H, W = support_feats_mp.shape
        k = v = support_feats_mp.reshape(B, C, H * W).permute(0, 2, 1)  # (B, 196, 1024)
        k = self.w_qk(k)  # (B, 196, 1024)

        # scaled dot-product attention
        if forward_novel:
            query_emb = self.query_embed_novel.weight
            q = query_emb.reshape(self.num_novel, self.num_queries, query_emb.size(-1))
        else:
            query_emb = self.query_embed.weight
            q = query_emb.reshape(self.num_base_cls, self.num_queries, query_emb.size(-1))

        q = q[support_gt_labels, ...]  # align with support_gt_labels
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
        weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)
        prototypes = torch.matmul(attn.softmax(-1), v)     # (B, 5, 1024)

        return weight, prototypes



@AGGREGATORS.register_module()
class PrototypesAssignment(BaseModule):
    def __init__(self, dim, num_bg=5):
        super().__init__()
        k_dim = dim // 4
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

        self.num_bg = num_bg
        if self.num_bg > 0:
            self.dummy = nn.Parameter(torch.Tensor(self.num_bg, dim))
            nn.init.normal_(self.dummy)
            self.linear = nn.Linear(dim, k_dim)
        self.gamma = nn.Parameter(torch.tensor(0.))
        self.fuse_mode = APA(channels = 1024)

    def forward(self, query_feature, prototypes, query_img_metas=None):
        """
        Args:
            query_feature: Tensor with shape (B, C, H, W)
            prototypes: Tensor with shape (num_supp, num_queries, C),
            query_img_metas: Visualization.
        Returns:
            class-specific query feature: tensor(B, C, H, W)
        """

        B, C, H, W = query_feature.shape
        num_supp, num_queries, _ = prototypes.shape
        q = query_feature.reshape(B, C, H*W).permute(0, 2, 1)   # (B, H*W, 1024)
        k = v = prototypes.reshape(num_supp * num_queries, C)

        q = self.w_qk(q)
        k = self.w_qk(k)

        if self.num_bg > 0:
            dummy_v = torch.zeros((self.num_bg, C), device='cuda')
            k = torch.cat([k, self.linear(self.dummy)], dim=0)
            v = torch.cat([v, dummy_v], dim=0)

        k = k.unsqueeze(0)
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.expand(B, -1, -1).transpose(-2, -1))
        attn.div_(0.5)

        out = torch.matmul(attn.softmax(-1), v)  # (B, 2850, 1024)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        # Without APA
        # out = query_feature + self.gamma * out

        # With APA
        out = self.fuse_mode(self.gamma * out, query_feature)
        
        return out

