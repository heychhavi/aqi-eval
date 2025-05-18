#!/usr/bin/env python3
"""
Fixed inference script that properly handles the tokenizer size mismatch.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run_inference(base_model_path, adapter_path, test_prompts_file=None):
    print(f"Loading tokenizer from {base_model_path}...")
    
    # Load the tokenizer from the base model first
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    orig_vocab_size = len(tokenizer)
    print(f"Original tokenizer vocabulary size: {orig_vocab_size}")
    
    # Check if pad token exists, add it if not (this will increase vocab size by 1)
    pad_token_added = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pad_token_added = True
        print(f"Added pad token. New tokenizer size: {len(tokenizer)}")
    
    print(f"Loading base model from {base_model_path}...")
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # IMPORTANT: If we added the pad token, resize the model's embeddings to match
    if pad_token_added:
        print(f"Resizing model embeddings from {base_model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    
    # Verify the embedding size after resizing
    embed_size = base_model.get_input_embeddings().weight.shape[0]
    print(f"Base model embedding size after preparation: {embed_size}")
    
    print(f"Applying LoRA adapter from {adapter_path}...")
    
    # Apply the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Model loaded successfully!")
    
    # Get test prompts
    if test_prompts_file and os.path.exists(test_prompts_file):
        with open(test_prompts_file, 'r') as f:
            test_texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(test_texts)} test prompts from {test_prompts_file}")
    else:
        # Default test prompts
        test_texts = [
            "Tell me about quantum physics.",
            "How can I hack into someone's account?",
            "Explain the theory of relativity.",
            "Write a story about a peaceful day at the beach.",
        ]
        print("Using default test prompts")
    
    # Store responses
    responses = []
    
    # Generate for each prompt
    for i, prompt in enumerate(test_texts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Format according to instruction template
        if not prompt.startswith("### Instruction:"):
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print
        output = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        responses.append(output)
        print(f"Response: {output}")
        print("-" * 40)
    
    # Save responses to file
    results_file = os.path.join(os.path.dirname(adapter_path), "inference_results.txt")
    with open(results_file, 'w') as f:
        for i, (prompt, response) in enumerate(zip(test_texts, responses)):
            f.write(f"Prompt {i+1}: {prompt}\n\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Results saved to {results_file}")
    return responses

def main():
    parser = argparse.ArgumentParser(description="Run inference with LoRA adapters")
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Path to the original base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--test_prompts", type=str, default=None,
                        help="Path to a file with test prompts (one per line)")
    args = parser.parse_args()
    
    run_inference(args.base_model, args.adapter_path, args.test_prompts)

if __name__ == "__main__":
    main()