"""
LLM Manager for handling DeepSeek model inference
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 2048
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         max_new_tokens: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        Generate a response using the LLM
        
        Args:
            prompt: User input
            context: Retrieved context from RAG
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        
        Returns:
            Generated response text
        """
        try:
            # Format the input with context if provided
            if context:
                formatted_input = f"Context: {context}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_input = f"User: {prompt}\nAssistant:"
            
            # Generate response
            response = self.pipeline(
                formatted_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract and clean the response
            generated_text = response[0]['generated_text']
            cleaned_response = self._clean_response(generated_text)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    def _clean_response(self, text: str) -> str:
        """
        Clean the generated response to remove unwanted artifacts
        """
        import re
        
        # Remove common prefixes and artifacts
        text = text.strip()
        
        # Remove reasoning markers and thinking patterns
        unwanted_patterns = [
            "<thinking>", "</thinking>",
            "<think>", "</think>",
            "<reasoning>", "</reasoning>",
            "Let me think", "I think",
            "User:", "Assistant:", "hBot:",
            "Step by step:", "First,", "Second,", "Third,",
            "In conclusion,", "To summarize,",
            "But wait,", "The user is asking",
            "Relevant information:", "So, the",
            "Maybe I should", "Yeah, that sounds good",
            "Let me just", "Alright,", "I'll go with that"
        ]
        
        for pattern in unwanted_patterns:
            text = text.replace(pattern, "")
        
        # Remove meta-commentary patterns using regex
        meta_patterns = [
            r"The assistant (?:provided|should|didn't).*?(?:\.|$)",
            r"But wait.*?(?:\.|$)",
            r"Relevant information:.*?(?:\.|$)", 
            r"So, the assistant.*?(?:\.|$)",
            r"The user is asking.*?(?:\.|$)",
            r"\*\*[^*]+:\*\*",  # Remove **bold headers:**
            r"Maybe I should.*?(?:\.|$)",
            r"I can respond with.*?(?:\.|$)",
            r"Let me (?:think|just|go).*?(?:\.|$)",
            r"Yeah, that sounds.*?(?:\.|$)",
            r"Alright,.*?(?:\.|$)",
            r"I'll go with.*?(?:\.|$)"
        ]
        
        for pattern in meta_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove repeated user/assistant exchanges
        text = re.sub(r"User:\s*.*?Assistant:\s*", "", text, flags=re.IGNORECASE)
        
        # Clean up multiple consecutive periods or question marks
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Extract the main response after cleaning
        # Look for the actual answer after all the meta-commentary
        lines = text.split('.')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like meta-commentary
            if (not line or 
                line.lower().startswith(('the assistant', 'but wait', 'relevant information', 
                                       'so, the', 'the user is', 'maybe i', 'let me', 
                                       'yeah,', 'alright', "i'll go", 'i can respond'))):
                continue
            cleaned_lines.append(line)
        
        if cleaned_lines:
            text = '. '.join(cleaned_lines)
            if not text.endswith('.') and not text.endswith('?') and not text.endswith('!'):
                text += '.'
        
        # Remove incomplete sentences at the end
        sentences = text.split('. ')
        if len(sentences) > 1 and len(sentences[-1]) < 10:
            text = '. '.join(sentences[:-1]) + '.'
        
        # Final cleanup - remove any remaining artifacts
        text = text.strip()
        
        # If the response is too short or seems corrupted, provide a fallback
        if len(text) < 20 or not text:
            return "I apologize, but I couldn't generate a clear response. Could you please rephrase your question?"
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "model_loaded": self.model is not None
        }
