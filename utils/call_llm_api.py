import os
from datetime import datetime
import json
import requests
import re
from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Try to import fix-busted-json, fallback to custom implementation if not available
try:
    from fix_busted_json import repair_json, first_json
    HAS_FIX_BUSTED_JSON = True
except ImportError:
    HAS_FIX_BUSTED_JSON = False
    print("Warning: fix-busted-json not installed. Using custom JSON repair implementation.")

# Read API keys from environment variables for better security
# Before running, ensure you have set the following environment variables:
# OPENROUTER_API_KEY, OPENAI_API_KEY
keys = {
    'or': os.getenv("OPENROUTER_API_KEY"),
    'oa': os.getenv("OPENAI_API_KEY"),
}

# development: economic use for development stage / final evaluation
# formal: ONLY evaluated in final stage
# optional: MAY evaluated in final stage, may not.
api_source_mapping = {
    # OpenAI-native models: provide both OA (direct) and OR (OpenRouter) routes
    "gpt41mini": {"oa": "gpt-4.1-mini-2025-04-14", "or": "openai/gpt-4.1-mini"},  #development-LLM
    "gpt41": {"oa": "gpt-4.1-2025-04-14", "or": "openai/gpt-4.1"}, #formal-LLM
    "o4mini": {"oa": "o4-mini-2025-04-16", "or": "openai/o4-mini"}, #formal-LRM 
    "gpt5": {"oa": "gpt-5", "or": "openai/gpt-5"}, #formal-LRM 
    "gpt5mini": {"oa": "gpt-5-mini-2025-08-07", "or": "openai/gpt-5-mini"}, #formal-LRM 

    # Non-OpenAI models:
    "gem25f": {"or": "google/gemini-2.5-flash"}, #development-LRM
    "gem25p": {"or": "google/gemini-2.5-pro"}, #formal-LRM

    "qwen3-235b": {"or": "qwen/qwen3-235b-a22b"}, #formal-LRM
    "qwq-32b": {"or": "qwen/qwq-32b"}, #development-LRM

    "dsv3": {"or": "deepseek/deepseek-chat-v3-0324"}, #formal-LLM
    "dsr1": {"or": "deepseek/deepseek-r1-0528"}, #formal-LRM

    "nemotron-ultra": {"or": "nvidia/llama-3.1-nemotron-ultra-253b-v1"}, #development-LRM
}

def resolve_model_and_source(model_name, keys):
    """
    Decide which provider to use based on key availability and model support,
    preferring 'oa' when available, else 'or'. For Nemotron, only 'nv' is allowed.
    Returns (api_source, provider_specific_model_id).
    """
    if model_name not in api_source_mapping:
        raise ValueError(f"Model '{model_name}' not found in api_source_mapping.")

    provider_map = api_source_mapping[model_name]

    # Prefer OA if key exists AND model supports OA mapping (non-None)
    if keys.get('oa') and provider_map.get('oa'):
        return 'oa', provider_map['oa']

    # Otherwise use OpenRouter if key exists and mapping available
    if keys.get('or') and provider_map.get('or'):
        return 'or', provider_map['or']

    # No usable key/mapping combination found
    raise ValueError(
        f"No available API key or supported mapping for model '{model_name}'. "
        f"Keys present: { {k: bool(v) for k, v in keys.items()} }; Supported sources: {list(provider_map.keys())}"
    )

def custom_repair_json(json_string):
    """
    Custom JSON repair implementation when fix-busted-json is not available.
    Handles common JSON malformation issues from LLMs.
    """
    if not json_string or not json_string.strip():
        return None, "Empty or None JSON string"
    
    # Step 1: Extract JSON from text
    start_idx = json_string.find('{')
    if start_idx == -1:
        return None, "No opening brace found"
    
    end_idx = json_string.rfind('}')
    if end_idx == -1:
        return None, "No closing brace found"
    
    json_string = json_string[start_idx:end_idx + 1]
    
    # Step 2: Try direct parsing first
    try:
        return json.loads(json_string), None
    except json.JSONDecodeError:
        pass
    
    # Step 3: Apply common fixes
    try:
        # Fix missing quotes around keys
        json_string = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_string)
        
        # Fix single quotes to double quotes
        json_string = re.sub(r"'([^']*)'", r'"\1"', json_string)
        
        # Fix trailing commas
        json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
        
        # Fix missing commas
        json_string = re.sub(r'"\s*"', r'", "', json_string)
        
        # Fix Python True/False/None
        json_string = json_string.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # Try parsing again
        return json.loads(json_string), None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to repair JSON: {str(e)}"


def robust_json_parse(response_text):
    """
    Robustly parse JSON from LLM response, handling malformed JSON.
    """
    if HAS_FIX_BUSTED_JSON:
        try:
            # Try to extract and repair JSON using fix-busted-json
            repaired_json = repair_json(response_text)
            return repaired_json, None
        except Exception as e:
            print(f"fix-busted-json failed: {e}")
            # Fallback to custom implementation
            return custom_repair_json(response_text)
    else:
        return custom_repair_json(response_text)


def safe_json_parse(response_text):
    """
    Safely parse JSON from response, with fallback to text extraction.
    """
    # First try to parse as regular JSON
    try:
        return json.loads(response_text), None
    except json.JSONDecodeError:
        pass
    
    # If that fails, try robust parsing
    parsed_data, error = robust_json_parse(response_text)
    if parsed_data:
        return parsed_data, None
    
    # If all parsing fails, return the raw text
    return response_text, f"JSON parsing failed: {error}"


def call_llm_api(messages, model_name, keys=keys, temperature=0.4, trial_info=None):
    """
    Calls a large language model API based on the specified model name.
    Now includes robust JSON parsing for malformed responses.
    """
    # Resolve provider and provider-specific model id based on key availability and mapping
    api_source, full_model_name = resolve_model_and_source(model_name, keys)

    trial_id = trial_info.get('trial_id', "unknown") if trial_info else "unknown"
    reasoning_content = None

    if api_source == "or":
        try:
            params = {
                "model": full_model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if model_name == "dsv31":
                params["reasoning"] = {"enabled": True} 

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {keys['or']}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(params)
            )
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                data = response.json()
                # Check if response has expected structure
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    reasoning_content = data['choices'][0]['message'].get('reasoning', None)
                    if reasoning_content is None:
                        reasoning_content = data['choices'][0]['message'].get('reasoning_content', None)
                    # Safely extract tokens with fallback
                    tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                elif 'content' in data:
                    # Alternative response format with direct 'content' key
                    content = data['content']
                    tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                elif 'response' in data:
                    # Another possible format
                    content = data['response']
                    tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                else:
                    # Response doesn't have expected structure, use robust parsing
                    print(f"[Trial {trial_id}] Response missing expected keys, using robust parsing")
                    print(f"[Trial {trial_id}] Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    content, tokens, method_used = extract_content_from_raw_response(response.text, trial_id)
                    print(f"[Trial {trial_id}] Content extracted using: {method_used}")
            except (json.JSONDecodeError, requests.exceptions.JSONDecodeError, KeyError) as e:
                print(f"[Trial {trial_id}] JSON decode error: {e}")
                print(f"[Trial {trial_id}] Response status: {response.status_code}, length: {len(response.text)} chars")
                
                # Save raw response to trial log file if trial_info is provided
                if trial_info and 'trial_dir' in trial_info:
                    trial_dir = trial_info['trial_dir']
                    trial_id_log = trial_info.get('trial_id', 'unknown') # Use a different variable name to avoid conflict
                    
                    raw_response_log_path = os.path.join(trial_dir, f"trial{trial_id_log}_raw_response.log")
                    
                    with open(raw_response_log_path, 'w', encoding='utf-8') as log_file:
                        log_file.write(f"Trial: {trial_id_log}\n")
                        log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        log_file.write(f"Error: {e}\n")
                        log_file.write(f"Response Status: {response.status_code}\n")
                        log_file.write(f"Response Length: {len(response.text)} characters\n")
                        log_file.write("-" * 80 + "\n")
                        log_file.write("RAW RESPONSE:\n")
                        log_file.write(response.text)
                        log_file.write("\n" + "-" * 80 + "\n")
                    
                    print(f"[Trial {trial_id_log}] Raw response saved to: {raw_response_log_path}")
                else:
                    print(f"[Trial {trial_id}] Raw response not saved (no trial_info provided)")
                
                # Use robust JSON parsing to extract content
                content, tokens, method_used = extract_content_from_raw_response(response.text, trial_id)
                print(f"[Trial {trial_id}] Content extracted using: {method_used}")
            
        except requests.exceptions.RequestException as e:
            print(f"[Trial {trial_id}] Request error: {e}")
            raise e
    
    elif api_source == "oa":
        try:
            client = OpenAI(api_key=keys['oa'])
            model_with_fix_temp = ["o4mini", "gpt5", "gpt5mini"] 
            if model_name in model_with_fix_temp:
                temperature = 1.0
            completion = client.chat.completions.create(
                model=full_model_name,
                messages=messages,
                temperature=temperature
            )
            content = completion.choices[0].message.content
            reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
            if reasoning_content is None:
                reasoning_content = getattr(completion.choices[0].message, 'reasoning', None)
            # Safely extract tokens with fallback
            if content is None:
                tokens = 0
            else:
                tokens = getattr(completion.usage, 'completion_tokens', len(content.split()))
        except Exception as e:
            print(f"[Trial {trial_id}] OA API error: {e}")
            raise e
    
    else:
        raise ValueError(f"Unknown API source: {api_source}")
    
    return content, reasoning_content, tokens


def extract_content_from_raw_response(raw_text, trial_id="unknown"):
    """
    Extract content from raw response using multiple strategies.
    Returns (content, tokens, method_used)
    """
    # Strategy 1: Try JSON repair
    try:
        parsed_data, error = safe_json_parse(raw_text)
        if parsed_data and isinstance(parsed_data, dict):
            if 'choices' in parsed_data and len(parsed_data['choices']) > 0:
                content = parsed_data['choices'][0]['message']['content']
                tokens = parsed_data.get('usage', {}).get('completion_tokens', len(content.split()))
                print(f"[Trial {trial_id}] ✓ JSON repair successful")
                return content, tokens, "JSON repair"
    except Exception as e:
        pass
    
    # Strategy 2: Try regex extraction with multiple patterns
    patterns = [
        r'"content":\s*"([^"]*)"',
        r'"content":\s*"([^"]*(?:\\"[^"]*)*)"',  # Handle escaped quotes
        r'"content":\s*"([^"]*)"[^}]*"completion_tokens":\s*(\d+)',
        r'"content":\s*"([^"]*)"[^}]*"total_tokens":\s*(\d+)'
    ]
    
    for i, pattern in enumerate(patterns):
        try:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                content = match.group(1)
                # Handle escaped quotes
                content = content.replace('\\"', '"')
                tokens = int(match.group(2)) if len(match.groups()) > 1 else len(content.split())
                print(f"[Trial {trial_id}] ✓ Regex extraction successful (pattern {i+1})")
                return content, tokens, f"Regex pattern {i+1}"
        except Exception as e:
            continue
    
    # Strategy 3: Manual parsing for common response formats
    try:
        # Look for content between quotes after "content":
        content_start = raw_text.find('"content":')
        if content_start != -1:
            # Find the opening quote
            quote_start = raw_text.find('"', content_start + 10)
            if quote_start != -1:
                # Find the closing quote, handling escaped quotes
                quote_end = quote_start + 1
                while quote_end < len(raw_text):
                    if raw_text[quote_end] == '"' and raw_text[quote_end-1] != '\\':
                        break
                    quote_end += 1
                
                if quote_end < len(raw_text):
                    content = raw_text[quote_start+1:quote_end]
                    content = content.replace('\\"', '"')
                    tokens = len(content.split())
                    print(f"[Trial {trial_id}] ✓ Manual parsing successful")
                    return content, tokens, "Manual parsing"
    except Exception as e:
        pass
    
    # Strategy 4: Fallback - return raw text
    print(f"[Trial {trial_id}] ⚠ All extraction methods failed, using raw text")
    return raw_text, len(raw_text.split()), "Raw text fallback"


# ===================================================================
# Main execution block to test all models sequentially
# ===================================================================
if __name__ == '__main__':
    sample_messages = [{"role": "user", "content": "Hi."}]

    print("--- Starting Model Tests ---")
    print("This will iterate through all defined models and test the ones with available API keys.\n")

    # Iterate through every model in the mapping
    for model_name in api_source_mapping:
        try:
            # For display purposes, resolve the source that was used.
            api_source, _ = resolve_model_and_source(model_name, keys)

            print(f"--- Testing model: {model_name} (Source: {api_source.upper()}) ---")
            # The call_llm_api function will automatically resolve the best API source
            # and raise an error if no valid API key is available for the model.
            content, reasoning_content, tokens = call_llm_api(sample_messages, model_name)
            print(f"✅ SUCCESS")
            print(f"Response: {content}")
            print(f"Response (Reason): {reasoning_content}")
            print(f"Tokens: {tokens}\n")
        except ValueError as e:
            # This can be triggered by resolve_model_and_source if no key is available.
            print(f"SKIPPING: {model_name}. Reason: {e}\n")
        except Exception as e:
            print(f"❌ FAILED: An error occurred while calling model {model_name}.")
            print(f"   Error details: {e}\n")

    print("--- All Model Tests Complete ---")

