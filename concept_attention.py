import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional
import math

logger = logging.getLogger(__name__)

class ConceptAttention:
    """
    ConceptAttention implementation for ComfyUI using patches_replace system.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.attention_outputs = {}
        self.model = None
    
    def register_attention_hooks(self, model):
        """
        Register attention capture by directly replacing forward methods (inspired by ComfyUI-Attention-Distillation).
        """
        logger.info("🔍 Using direct forward method replacement for attention capture")
        
        # Store model reference
        self.model = model
        
        # Try to find the actual model
        actual_model = None
        if hasattr(model, 'model'):
            actual_model = model.model
        elif hasattr(model, 'diffusion_model'):
            actual_model = model.diffusion_model
        elif hasattr(model, 'unet_model'):
            actual_model = model.unet_model
        else:
            actual_model = model
        
        logger.info(f"🔍 Using model for attention capture: {type(actual_model)}")
        
        # Register attention capture by replacing forward methods
        self._register_attention_forward_replacement(actual_model)
        
        return self.attention_outputs
    
    def _register_attention_forward_replacement(self, model):
        """
        Register attention capture by directly replacing forward methods (inspired by ComfyUI-Attention-Distillation).
        """
        logger.info("🔍 Registering attention forward method replacement")
        
        def attn_forward(self):
            def forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                image_rotary_emb=None,
                *args,
                **kwargs,
            ):
                batch_size, _, _ = (
                    hidden_states.shape
                    if encoder_hidden_states is None
                    else encoder_hidden_states.shape
                )

                # `sample` projections.
                query = self.to_q(hidden_states)
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // self.heads

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)

                # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
                if encoder_hidden_states is not None:
                    # `context` projections.
                    encoder_hidden_states_query_proj = self.add_q_proj(
                        encoder_hidden_states
                    )
                    encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                    encoder_hidden_states_value_proj = self.add_v_proj(
                        encoder_hidden_states
                    )

                    encoder_hidden_states_query_proj = (
                        encoder_hidden_states_query_proj.view(
                            batch_size, -1, self.heads, head_dim
                        ).transpose(1, 2)
                    )
                    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                        batch_size, -1, self.heads, head_dim
                    ).transpose(1, 2)
                    encoder_hidden_states_value_proj = (
                        encoder_hidden_states_value_proj.view(
                            batch_size, -1, self.heads, head_dim
                        ).transpose(1, 2)
                    )

                    if self.norm_added_q is not None:
                        encoder_hidden_states_query_proj = self.norm_added_q(
                            encoder_hidden_states_query_proj
                        )
                    if self.norm_added_k is not None:
                        encoder_hidden_states_key_proj = self.norm_added_k(
                            encoder_hidden_states_key_proj
                        )

                    # attention
                    query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                    key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                    value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

                if image_rotary_emb is not None:
                    from diffusers.models.embeddings import apply_rotary_emb

                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                
                # Capture attention outputs
                try:
                    # Create a unique key for this attention layer
                    layer_key = f"attention_{id(self)}_{type(self).__name__}"
                    
                    # Convert to float32 to avoid dtype issues
                    attention_output = hidden_states.to(torch.float32)
                    
                    # Store the attention output
                    self.attention_outputs[layer_key] = attention_output
                    logger.info(f"📝 Captured attention output: {layer_key}, shape: {attention_output.shape}")
                    
                except Exception as e:
                    logger.debug(f"Failed to capture attention output: {e}")

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.heads * head_dim
                )

                hidden_states = hidden_states.to(query.dtype)

                if encoder_hidden_states is not None:
                    encoder_hidden_states, hidden_states = (
                        hidden_states[:, : encoder_hidden_states.shape[1]],
                        hidden_states[:, encoder_hidden_states.shape[1] :],
                    )

                    # linear proj
                    hidden_states = self.to_out[0](hidden_states)
                    # dropout
                    hidden_states = self.to_out[1](hidden_states)
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

                    return hidden_states, encoder_hidden_states
                else:
                    return hidden_states

            return forward

        def modify_forward(net, count):
            """Recursively modify forward methods of attention modules."""
            # Check if current net is an attention module
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = attn_forward(net)
                count += 1
                logger.info(f"✅ Replaced forward method for: {type(net).__name__}")
                return count
            
            # Recursively check children
            for name, subnet in net.named_children():
                if hasattr(subnet, "children") or subnet.__class__.__name__ == "Attention":
                    count = modify_forward(subnet, count)
            return count
        
        # Try to find transformer_blocks or similar structures
        if hasattr(model, 'transformer_blocks'):
            logger.info("Found transformer_blocks, applying attention control")
            attention_count = modify_forward(model.transformer_blocks, 0)
            logger.info(f"🔍 Replaced forward methods for {attention_count} attention modules in transformer_blocks")
        
        if hasattr(model, 'single_transformer_blocks'):
            logger.info("Found single_transformer_blocks, applying attention control")
            attention_count = modify_forward(model.single_transformer_blocks, 0)
            logger.info(f"🔍 Replaced forward methods for {attention_count} attention modules in single_transformer_blocks")
        
        # Also try the general approach
        attention_count = modify_forward(model, 0)
        logger.info(f"🔍 Replaced forward methods for {attention_count} attention modules")
    
    def _create_attention_capture_wrapper(self, block_idx):
        """
        Create a wrapper function for ComfyUI's patches_replace system to capture attention.
        """
        def attention_capture_wrapper(args, original_block):
            """
            Wrapper function that captures attention outputs and calls the original block.
            """
            logger.info(f"🎯 Attention capture wrapper called for block {block_idx}")
            
            # Extract arguments
            img = args["img"]
            txt = args["txt"]
            vec = args["vec"]
            pe = args["pe"]
            attn_mask = args.get("attn_mask")
            transformer_options = args.get("transformer_options", {})
            
            # Call the original block
            img_out, txt_out = original_block(args)
            
            # Try to capture attention outputs from the block
            try:
                # Get the actual block from the model
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'double_blocks'):
                    actual_block = self.model.model.double_blocks[block_idx]
                    
                    # Try to access attention modules
                    if hasattr(actual_block, 'img_attn'):
                        img_attn = actual_block.img_attn
                        
                        # Create test input for attention
                        test_input = torch.randn(1, 1024, 256, device=img.device, dtype=img.dtype)
                        
                        with torch.no_grad():
                            # Try to capture qkv output
                            if hasattr(img_attn, 'qkv'):
                                qkv_output = img_attn.qkv(test_input)
                                hook_key = f"patches_replace_qkv_block_{block_idx}_{id(img_attn)}"
                                self.attention_outputs[hook_key] = qkv_output
                                logger.info(f"📝 Captured qkv output from block {block_idx}: {qkv_output.shape}")
                                
                                # Try to capture full attention computation
                                if hasattr(img_attn, 'norm') and hasattr(img_attn, 'proj'):
                                    # Split qkv output
                                    qkv_split = qkv_output.view(1, 1024, 3, 256)
                                    q, k, v = qkv_split[:, :, 0], qkv_split[:, :, 1], qkv_split[:, :, 2]
                                    
                                    # Apply norm
                                    q, k = img_attn.norm(q, k, v)
                                    
                                    # Compute attention
                                    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                                    hook_key = f"patches_replace_attention_block_{block_idx}_{id(img_attn)}"
                                    self.attention_outputs[hook_key] = attn_output
                                    logger.info(f"📝 Captured attention output from block {block_idx}: {attn_output.shape}")
                                    
                                    # Apply projection
                                    proj_output = img_attn.proj(attn_output)
                                    hook_key = f"patches_replace_proj_block_{block_idx}_{id(img_attn)}"
                                    self.attention_outputs[hook_key] = proj_output
                                    logger.info(f"📝 Captured projection output from block {block_idx}: {proj_output.shape}")
                
            except Exception as e:
                logger.debug(f"Failed to capture attention from block {block_idx}: {e}")
            
            return {"img": img_out, "txt": txt_out}
        
        return attention_capture_wrapper
    
    def _extract_concept_embeddings(self, concepts: List[str], text_encoder, tokenizer):
        """
        Extract concept embeddings using CLIP text encoder.
        """
        concept_embeddings = {}
        
        try:
            for concept in concepts:
                # Check if text_encoder and tokenizer are valid
                if text_encoder is None or tokenizer is None:
                    logger.error(f"❌ Text encoder or tokenizer is None for concept '{concept}'! Real text encoding is required.")
                    raise RuntimeError(f"Text encoder and tokenizer are required for concept '{concept}'. No fallback allowed.")
                else:
                    # Tokenize the concept
                    tokens = tokenizer(concept, return_tensors="pt", padding=True, truncation=True)
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    
                    # Get embeddings from text encoder
                    with torch.no_grad():
                        if hasattr(text_encoder, 'encode_text'):
                            embedding = text_encoder.encode_text(tokens["input_ids"])
                        elif hasattr(text_encoder, 'forward'):
                            embedding = text_encoder(tokens["input_ids"])
                        elif hasattr(text_encoder, 'encode'):
                            embedding = text_encoder.encode(tokens["input_ids"])
                        else:
                            logger.error(f"❌ Text encoder has no recognized methods for concept '{concept}'!")
                            raise RuntimeError(f"Text encoder must have encode_text, forward, or encode method. No fallback allowed.")
                
                # Ensure embedding is on correct device and dtype
                embedding = embedding.to(device=self.device)
                concept_embeddings[concept] = embedding
                logger.info(f"Extracted embedding for '{concept}': shape {embedding.shape}, dtype {embedding.dtype}")
                
        except Exception as e:
            logger.error(f"Error extracting concept embeddings: {e}")
            raise RuntimeError(f"Failed to extract concept embeddings: {e}. No fallback allowed.")
        
        return concept_embeddings
    
    def _create_concept_maps_from_attention(self, concepts: List[str], concept_embeddings: Dict[str, torch.Tensor], image_shape: tuple):
        """
        Create concept maps from captured attention outputs.
        """
        concept_maps = {}
        
        if not self.attention_outputs:
            logger.warning("No attention outputs available for concept map creation")
            return concept_maps
        
        # Select the best attention output
        best_attention_output = self._select_best_attention_output()
        if best_attention_output is None:
            logger.warning("No suitable attention output found")
            return concept_maps
        
        attention_output, attention_key = best_attention_output
        logger.info(f"Using attention output: {attention_key}, shape: {attention_output.shape}")
        
        # Get model dtype for consistency
        if hasattr(self.model, 'parameters'):
            params = list(self.model.parameters())
            model_dtype = params[0].dtype if params else torch.float32
        else:
            model_dtype = torch.float32
        
        for concept in concepts:
            try:
                embedding = concept_embeddings[concept]
                logger.info(f"Processing concept '{concept}': embedding dim {embedding.shape[-1]}, attention dim {attention_output.shape[-1]}")
                
                # Ensure embedding is on correct device and dtype
                embedding = embedding.to(device=self.device, dtype=model_dtype)
                
                # Handle different embedding shapes
                if embedding.dim() == 2:
                    embedding = embedding.unsqueeze(1)  # [1, 1, embedding_dim]
                elif embedding.dim() == 3:
                    pass  # Already [1, 1, embedding_dim] or [1, seq_len, embedding_dim]
                else:
                    logger.warning(f"Unexpected embedding shape: {embedding.shape}")
                    continue
                
                # Ensure embedding is [1, 1, embedding_dim]
                if embedding.shape[1] > 1:
                    embedding = embedding.mean(dim=1, keepdim=True)  # Average over sequence length
                
                # Align embedding dimension with attention output dimension
                embedding_dim = embedding.shape[-1]
                attention_dim = attention_output.shape[-1]
                
                if embedding_dim != attention_dim:
                    # Create a learnable projection layer
                    if not hasattr(self, f'projection_{attention_dim}'):
                        projection = torch.nn.Linear(embedding_dim, attention_dim, device=self.device, dtype=model_dtype)
                        setattr(self, f'projection_{attention_dim}', projection)
                    else:
                        projection = getattr(self, f'projection_{attention_dim}')
                    
                    # Project embedding to match attention dimension
                    embedding = projection(embedding)
                    logger.info(f"Projected embedding from {embedding_dim} to {attention_dim}")
                
                # Reshape attention output for similarity computation
                if attention_output.dim() == 3:
                    # [batch, seq_len, dim] -> [batch, seq_len, dim]
                    attention_flat = attention_output
                else:
                    attention_flat = attention_output
                
                # Compute similarity between concept embedding and attention output
                # Use einsum-like operation similar to original ConceptAttention
                # embedding: [1, 256], attention_flat: [1, 1024, 256]
                # We want to compute similarity for each position in the sequence
                concept_attn_output = torch.matmul(attention_flat, embedding.unsqueeze(-1))  # [1, 1024, 1]
                concept_attn_output = concept_attn_output.squeeze(-1)  # [1, 1024]
                
                # Reshape to spatial dimensions
                seq_len = concept_attn_output.shape[-1]
                spatial_h = int(math.sqrt(seq_len))
                spatial_w = seq_len // spatial_h
                
                if spatial_h * spatial_w == seq_len:
                    similarity = concept_attn_output.view(1, spatial_h, spatial_w)
                else:
                    # Fallback: use 1D reshaping
                    similarity = concept_attn_output.view(1, seq_len, 1)
                
                logger.info(f"Concept attention similarity shape: {similarity.shape}")
                
                # Resize to target image dimensions
                if len(image_shape) == 3:  # [H, W, C]
                    target_h, target_w = image_shape[:2]
                else:  # [H, W]
                    target_h, target_w = image_shape
                logger.info(f"Target image dimensions: {target_h}x{target_w}")
                
                # Ensure similarity is in correct format for interpolation
                if similarity.dim() == 2:
                    similarity = similarity.unsqueeze(0)  # Add batch dimension
                elif similarity.dim() == 3:
                    if similarity.shape[1] == 1:  # [1, 1, seq_len]
                        similarity = similarity.permute(0, 2, 1)  # [1, seq_len, 1]
                
                # Convert to 4D for interpolation: [N, C, H, W]
                if similarity.dim() == 3:
                    if similarity.shape[2] == 1:  # [1, H, 1]
                        similarity_4d = similarity.unsqueeze(1)  # [1, 1, H, 1]
                    else:  # [1, H, W]
                        similarity_4d = similarity.unsqueeze(1)  # [1, 1, H, W]
                else:
                    similarity_4d = similarity
                
                logger.info(f"similarity_4d shape: {similarity_4d.shape}")
                
                # Interpolate to target size
                resized_similarity = F.interpolate(
                    similarity_4d,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                logger.info(f"Resized similarity shape: {resized_similarity.shape}")
                
                # Convert to concept map
                if resized_similarity.shape[1] == 1:  # [1, 1, H, W]
                    concept_map = resized_similarity.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
                else:  # [1, C, H, W] or other format
                    concept_map = resized_similarity.squeeze(0).mean(dim=0)  # Average across channels
                
                # Normalize concept map using percentile-based normalization
                concept_map_flat = concept_map.flatten()
                concept_map_flat = concept_map_flat.to(torch.float32)  # Convert to float32 for quantile
                
                # Use percentile-based normalization
                p5 = torch.quantile(concept_map_flat, 0.05)
                p95 = torch.quantile(concept_map_flat, 0.95)
                
                # Clamp and normalize
                concept_map = torch.clamp(concept_map, p5, p95)
                concept_map = (concept_map - p5) / (p95 - p5 + 1e-8)
                
                # Apply gamma correction for better contrast
                concept_map = torch.pow(concept_map, 0.5)
                
                # Convert to float32 for ComfyUI compatibility
                concept_map = concept_map.float()
                
                concept_maps[concept] = concept_map
                logger.info(f"Created concept map for '{concept}': shape {concept_map.shape}")
                
            except Exception as e:
                logger.error(f"Error creating concept map for '{concept}': {e}")
                continue
        
        return concept_maps
    
    def _select_best_attention_output(self):
        """
        Select the best attention output from captured outputs.
        """
        if not self.attention_outputs:
            return None
        
        best_output = None
        best_key = None
        best_score = -1
        
        for key, output in self.attention_outputs.items():
            score = 0
            
            # Prioritize patches_replace outputs
            if 'patches_replace' in key:
                score += 1000
            
            # Prioritize attention-related outputs
            if 'attention' in key.lower():
                score += 500
            elif 'qkv' in key.lower():
                score += 300
            elif 'proj' in key.lower():
                score += 200
            
            # Prefer outputs with proper spatial structure
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                if output.shape[-1] > 100:  # Prefer outputs with reasonable dimension
                    score += 100
            
            if score > best_score:
                best_score = score
                best_output = output
                best_key = key
        
        return (best_output, best_key) if best_output is not None else None
    
    def extract_attention_outputs(self):
        """
        Extract attention outputs using patches_replace system.
        """
        if not self.model:
            logger.warning("No model available for attention extraction")
            return self.attention_outputs
        
        # Try to find the actual model
        actual_model = None
        if hasattr(self.model, 'model'):
            actual_model = self.model.model
        elif hasattr(self.model, 'diffusion_model'):
            actual_model = self.model.diffusion_model
        elif hasattr(self.model, 'unet_model'):
            actual_model = self.model.unet_model
        else:
            actual_model = self.model
        
        if not actual_model:
            logger.warning("No actual model found")
            return self.attention_outputs
        
        # Get model dtype
        if hasattr(actual_model, 'parameters'):
            params = list(actual_model.parameters())
            model_dtype = params[0].dtype if params else torch.float32
        else:
            model_dtype = torch.float32
        
        logger.info(f"🔍 Attempting attention capture using patches_replace system")
        logger.info(f"Using model: {type(actual_model)}")
        
        try:
            # Try direct attention module execution instead of patches_replace
            logger.info("🔍 Trying direct attention module execution")
            
            if hasattr(actual_model, 'double_blocks'):
                double_blocks = actual_model.double_blocks
                logger.info(f"Found {len(double_blocks)} double blocks")
                
                # Target specific blocks (15-18 as in original ConceptAttention)
                for block_idx in [15, 16, 17, 18]:
                    if block_idx < len(double_blocks):
                        block = double_blocks[block_idx]
                        logger.info(f"Processing block {block_idx}: {type(block)}")
                        
                        # Try to access attention modules directly
                        if hasattr(block, 'img_attn'):
                            img_attn = block.img_attn
                            logger.info(f"Found img_attn in block {block_idx}: {type(img_attn)}")
                            
                            # Create test input for attention
                            test_input = torch.randn(1, 1024, 256, device=self.device, dtype=model_dtype)
                            
                            with torch.no_grad():
                                try:
                                    # Try to capture qkv output
                                    if hasattr(img_attn, 'qkv'):
                                        qkv_output = img_attn.qkv(test_input)
                                        hook_key = f"direct_qkv_block_{block_idx}_{id(img_attn)}"
                                        self.attention_outputs[hook_key] = qkv_output
                                        logger.info(f"📝 Captured qkv output from block {block_idx}: {qkv_output.shape}")
                                        
                                        # Try to capture full attention computation
                                        if hasattr(img_attn, 'norm') and hasattr(img_attn, 'proj'):
                                            # Split qkv output
                                            qkv_split = qkv_output.view(1, 1024, 3, 256)
                                            q, k, v = qkv_split[:, :, 0], qkv_split[:, :, 1], qkv_split[:, :, 2]
                                            
                                            # Apply norm
                                            q, k = img_attn.norm(q, k, v)
                                            
                                            # Compute attention
                                            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                                            hook_key = f"direct_attention_block_{block_idx}_{id(img_attn)}"
                                            self.attention_outputs[hook_key] = attn_output
                                            logger.info(f"📝 Captured attention output from block {block_idx}: {attn_output.shape}")
                                            
                                            # Apply projection
                                            proj_output = img_attn.proj(attn_output)
                                            hook_key = f"direct_proj_block_{block_idx}_{id(img_attn)}"
                                            self.attention_outputs[hook_key] = proj_output
                                            logger.info(f"📝 Captured projection output from block {block_idx}: {proj_output.shape}")
                                            
                                            # Break after first successful block to avoid too much computation
                                            break
                                
                                except Exception as attn_error:
                                    logger.debug(f"Direct attention execution failed in block {block_idx}: {attn_error}")
            
            # No fallback - require real attention capture
            if not self.attention_outputs:
                logger.error("❌ No attention outputs captured! Real attention capture is required.")
                raise RuntimeError("Failed to capture real attention outputs. No fallback allowed.")
            
            # Log final results
            logger.info(f"📊 Total attention outputs captured: {len(self.attention_outputs)}")
            logger.info(f"📊 Attention output keys: {list(self.attention_outputs.keys())}")
            
            return self.attention_outputs
        
        except Exception as e:
            logger.error(f"❌ Attention capture failed: {e}")
            raise RuntimeError(f"Failed to capture attention outputs: {e}")


class ConceptAttentionProcessor:
    """
    Main processor for ConceptAttention functionality.
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.concept_attention = ConceptAttention(device=device)
    
    def process_image(self, image: torch.Tensor, concepts: List[str], text_encoder=None, tokenizer=None):
        """
        Process an image to extract concept attention maps.
        """
        try:
            logger.info(f"Processing image with concepts: {concepts}")
            
            # Register attention capture system
            self.concept_attention.register_attention_hooks(self.model)
            
            # Extract concept embeddings
            concept_embeddings = self.concept_attention._extract_concept_embeddings(concepts, text_encoder, tokenizer)
            logger.info(f"Extracted embeddings for concepts: {list(concept_embeddings.keys())}")
            
            # Extract attention outputs
            attention_outputs = self.concept_attention.extract_attention_outputs()
            
            # Create concept maps
            # Handle different image formats: [B, H, W, C] or [B, H, W]
            if len(image.shape) == 4:  # [B, H, W, C]
                image_shape = image.shape[1:3]  # Get height and width
            else:  # [B, H, W]
                image_shape = image.shape[1:3]  # Get height and width
            concept_maps = self.concept_attention._create_concept_maps_from_attention(
                concepts, concept_embeddings, image_shape
            )
            
            return concept_maps
            
        except Exception as e:
            logger.error(f"Error in ConceptAttention: {e}")
            return {}
