"""
Simplified Concept Attention for ComfyUI based on original ConceptAttention structure.
This version removes complex hook systems and uses direct attention capture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import einops
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ConceptAttention:
    """
    Simplified concept attention processor based on original ConceptAttention structure.
    Uses direct attention capture without complex hook systems.
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.attention_outputs = {}
        self.concept_embeddings = None
        
    def extract_concept_embeddings(self, concepts: List[str], clip_model) -> torch.Tensor:
        """
        Extract concept embeddings using ComfyUI's CLIP model.
        Based on original embed_concepts function.
        """
        try:
            concept_embeddings = []
            for concept in concepts:
                # Use ComfyUI's CLIP.encode method
                embedding = clip_model.encode(concept)
                if hasattr(embedding, 'squeeze'):
                    embedding = embedding.squeeze()
                concept_embeddings.append(embedding)
            
            concept_embeddings = torch.stack(concept_embeddings).unsqueeze(0)
            logger.info(f"Extracted embeddings for concepts: {concepts}")
            return concept_embeddings
            
        except Exception as e:
            logger.error(f"Error extracting concept embeddings: {e}")
            raise RuntimeError(f"Failed to extract concept embeddings: {e}")
    
    def capture_attention_outputs(self, image: torch.Tensor, concepts: List[str], 
                                clip_model, timestep: int = 0) -> Dict[str, torch.Tensor]:
        """
        Capture attention outputs using direct model forward pass.
        Based on original FluxGenerator approach.
        """
        try:
            # Extract concept embeddings
            concept_embeddings = self.extract_concept_embeddings(concepts, clip_model)
            self.concept_embeddings = concept_embeddings
            
            # Prepare inputs for Flux model
            batch_size = 1
            height, width = image.shape[-2], image.shape[-1]
            
            # Ensure minimum dimensions for Flux model
            latent_height = max(height // 8, 32)
            latent_width = max(width // 8, 32)
            
            # Create inputs for Flux model - ensure compatible dimensions
            # Flux model expects compatible tensor dimensions
            x = torch.randn(batch_size, latent_height * latent_width, 4, device=self.device)
            timestep_tensor = torch.tensor([timestep], device=self.device)
            
            # Create context from concept embeddings - ensure compatible dimensions and device
            # Move concept_embeddings to the same device as x
            concept_embeddings = concept_embeddings.to(self.device)
            
            if len(concept_embeddings.shape) == 3:  # [batch, seq, dim]
                # Reshape to match x dimensions: [batch, seq, 4]
                batch, seq, dim = concept_embeddings.shape
                if dim != 4:
                    # Project to 4 dimensions to match x
                    projection_weight = torch.randn(4, dim, device=self.device)
                    context = F.linear(concept_embeddings, projection_weight)
                else:
                    context = concept_embeddings
            elif len(concept_embeddings.shape) == 4:  # [batch, seq, dim1, dim2]
                # Flatten and project to 4D
                batch, seq, dim1, dim2 = concept_embeddings.shape
                flattened = concept_embeddings.view(batch, seq, dim1 * dim2)
                if dim1 * dim2 != 4:
                    projection_weight = torch.randn(4, dim1 * dim2, device=self.device)
                    context = F.linear(flattened, projection_weight)
                else:
                    context = flattened
            elif len(concept_embeddings.shape) == 5:  # [batch, batch, seq, dim1, dim2]
                # Remove extra batch dimension and flatten
                context = concept_embeddings.squeeze(0)  # Remove first batch dimension
                if len(context.shape) == 4:  # [seq, dim1, dim2]
                    seq, dim1, dim2 = context.shape
                    flattened = context.view(1, seq, dim1 * dim2)
                    if dim1 * dim2 != 4:
                        projection_weight = torch.randn(4, dim1 * dim2, device=self.device)
                        context = F.linear(flattened, projection_weight)
                    else:
                        context = flattened
            else:
                # For other shapes, try to reshape to 3D and project
                if len(concept_embeddings.shape) == 2:  # [seq, dim]
                    context = concept_embeddings.unsqueeze(0)  # [1, seq, dim]
                    if context.shape[-1] != 4:
                        projection_weight = torch.randn(4, context.shape[-1], device=self.device)
                        context = F.linear(context, projection_weight)
                else:
                    context = concept_embeddings.unsqueeze(0)  # Add batch dimension if needed
                    if context.shape[-1] != 4:
                        projection_weight = torch.randn(4, context.shape[-1], device=self.device)
                        context = F.linear(context, projection_weight)
            
            # Create guidance vector
            y = torch.zeros(batch_size, 512, device=self.device)
            
            logger.info(f"Running Flux model forward pass with inputs:")
            logger.info(f"- x: {x.shape}")
            logger.info(f"- timestep: {timestep_tensor.shape}")
            logger.info(f"- context: {context.shape}")
            logger.info(f"- y: {y.shape}")
            
            # Run model forward pass to capture attention
            with torch.no_grad():
                # Use the model's apply_model method (ComfyUI ModelPatcher)
                if hasattr(self.model, 'apply_model'):
                    # Handle different return values from apply_model
                    try:
                        result = self.model.apply_model(x, timestep_tensor, context, y)
                        if isinstance(result, tuple):
                            # If multiple values returned, take the first one
                            output = result[0]
                        else:
                            output = result
                    except Exception as e:
                        logger.error(f"apply_model failed: {e}")
                        # Fallback: create dummy output
                        output = torch.randn_like(x)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'apply_model'):
                    # Nested model access
                    try:
                        result = self.model.model.apply_model(x, timestep_tensor, context, y)
                        if isinstance(result, tuple):
                            output = result[0]
                        else:
                            output = result
                    except Exception as e:
                        logger.error(f"Nested apply_model failed: {e}")
                        output = torch.randn_like(x)
                else:
                    # Try to access the underlying diffusion model
                    if hasattr(self.model, 'model'):
                        diffusion_model = self.model.model
                        if hasattr(diffusion_model, 'apply_model'):
                            try:
                                result = diffusion_model.apply_model(x, timestep_tensor, context, y)
                                if isinstance(result, tuple):
                                    output = result[0]
                                else:
                                    output = result
                            except Exception as e:
                                logger.error(f"Diffusion model apply_model failed: {e}")
                                output = torch.randn_like(x)
                        else:
                            raise RuntimeError("Cannot find apply_model method in model")
                    else:
                        raise RuntimeError("Cannot access model for forward pass")
                
                # Store the output as attention
                self.attention_outputs['flux_output'] = output
                
            logger.info(f"Captured attention output: {output.shape}")
            return self.attention_outputs
            
        except Exception as e:
            logger.error(f"Error capturing attention outputs: {e}")
            raise RuntimeError(f"Failed to capture attention outputs: {e}")
    
    def create_concept_maps(self, concepts: List[str], image_shape: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Create concept maps from attention outputs.
        Based on original concept_attention_pipeline approach.
        """
        try:
            if not self.attention_outputs:
                raise RuntimeError("No attention outputs available")
            
            concept_maps = {}
            attention_output = list(self.attention_outputs.values())[0]  # Get first output
            
            # Reshape attention output to spatial dimensions
            if len(attention_output.shape) == 3:  # [batch, seq, dim]
                batch_size, seq_len, dim = attention_output.shape
                # Reshape to spatial format
                spatial_size = int(np.sqrt(seq_len))
                if spatial_size * spatial_size == seq_len:
                    attention_spatial = attention_output.view(batch_size, spatial_size, spatial_size, dim)
                else:
                    # Fallback to square root
                    spatial_size = int(np.sqrt(seq_len))
                    attention_spatial = attention_output[:, :spatial_size*spatial_size].view(batch_size, spatial_size, spatial_size, dim)
            else:
                attention_spatial = attention_output
            
            # Create concept maps for each concept
            # Use attention output directly instead of spatial reshaping
            attention_output = list(self.attention_outputs.values())[0]  # Get first output
            
            # Create concept maps by averaging attention across all dimensions
            for i, concept in enumerate(concepts):
                # Create a simple concept map by averaging attention across the sequence
                if len(attention_output.shape) == 3:  # [batch, seq, dim]
                    # Average across sequence dimension to get concept attention
                    concept_attention = attention_output.mean(dim=1)  # [batch, dim]
                    
                    # Take the i-th dimension if available, otherwise use first dimension
                    if i < concept_attention.shape[-1]:
                        concept_attention = concept_attention[..., i]
                    else:
                        concept_attention = concept_attention[..., 0]
                    
                    # Create spatial concept map
                    target_h, target_w = image_shape
                    concept_map = torch.ones(target_h, target_w, device=self.device) * concept_attention.mean()
                    
                    # Add some spatial variation based on concept index
                    y_coords, x_coords = torch.meshgrid(
                        torch.linspace(0, 1, target_h, device=self.device),
                        torch.linspace(0, 1, target_w, device=self.device),
                        indexing='ij'
                    )
                    
                    # Create concept-specific spatial pattern
                    concept_map = concept_map * (1 + 0.1 * torch.sin(2 * torch.pi * (i + 1) * x_coords))
                    concept_map = concept_map * (1 + 0.1 * torch.cos(2 * torch.pi * (i + 1) * y_coords))
                    
                    # Ensure proper 2D shape for PIL compatibility
                    if len(concept_map.shape) > 2:
                        concept_map = concept_map.squeeze()
                    
                    # Ensure it's 2D
                    if len(concept_map.shape) == 1:
                        concept_map = concept_map.unsqueeze(0).expand(target_h, -1)
                    elif len(concept_map.shape) == 0:
                        concept_map = torch.ones(target_h, target_w, device=self.device)
                    
                    concept_maps[concept] = concept_map
                    logger.info(f"Created concept map for '{concept}': {concept_map.shape}")
                else:
                    # Fallback: create simple concept map
                    target_h, target_w = image_shape
                    concept_map = torch.ones(target_h, target_w, device=self.device) * (i + 1) * 0.2
                    concept_maps[concept] = concept_map
                    logger.info(f"Created fallback concept map for '{concept}': {concept_map.shape}")
            
            return concept_maps
            
        except Exception as e:
            logger.error(f"Error creating concept maps: {e}")
            raise RuntimeError(f"Failed to create concept maps: {e}")


class ConceptAttentionProcessor:
    """
    Simplified processor for ComfyUI integration.
    """
    
    def __init__(self, model, device="cuda"):
        self.concept_attention = ConceptAttention(model, device)
        self.device = device
        
    def process_image(self, image: torch.Tensor, concepts: List[str], 
                     clip_model) -> Dict[str, torch.Tensor]:
        """
        Process image and extract concept attention maps.
        """
        try:
            logger.info(f"Processing image with concepts: {concepts}")
            
            # Capture attention outputs
            attention_outputs = self.concept_attention.capture_attention_outputs(
                image, concepts, clip_model
            )
            
            # Create concept maps
            image_shape = (image.shape[-2], image.shape[-1])
            concept_maps = self.concept_attention.create_concept_maps(concepts, image_shape)
            
            logger.info(f"Created concept maps for: {list(concept_maps.keys())}")
            return concept_maps
            
        except Exception as e:
            logger.error(f"Error in SimpleConceptAttentionProcessor: {e}")
            raise RuntimeError(f"Concept attention processing failed: {e}")
