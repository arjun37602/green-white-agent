#!/usr/bin/env python3
"""
Extract conversations from JSONL results file.

Usage:
    python extract_conversations.py <jsonl_file_path>
"""

import json
import sys
from pathlib import Path

def clean_message(msg):
    """Clean up a message by removing unnecessary fields."""
    cleaned = {
        'iteration': msg.get('iteration'),
        'timestamp': msg.get('timestamp')
    }
    
    # Handle user messages
    if 'user' in msg:
        cleaned['user'] = msg['user']
    
    # Handle assistant messages - extract only essential fields
    if 'assistant' in msg:
        assistant = msg['assistant']
        if 'result' in assistant and 'message' in assistant['result']:
            message = assistant['result']['message']
            # Extract parts and parse the JSON text content
            parts = message.get('parts', [])
            cleaned_parts = []
            
            for part in parts:
                if part.get('kind') == 'text' and 'text' in part:
                    # Try to parse the text as JSON and extract analysis
                    try:
                        text_json = json.loads(part['text'])
                        cleaned_parts.append({
                            'analysis': text_json.get('analysis', ''),
                            'plan': text_json.get('plan', ''),
                            'commands': text_json.get('commands', []),
                            'task_complete': text_json.get('task_complete', False)
                        })
                    except (json.JSONDecodeError, TypeError):
                        # If parsing fails, try to extract from markdown code block
                        text = part['text']
                        if '```json' in text or '```' in text:
                            # Extract JSON from markdown code block
                            try:
                                # Remove markdown code block markers
                                json_text = text.replace('```json', '').replace('```', '').strip()
                                text_json = json.loads(json_text)
                                cleaned_parts.append({
                                    'analysis': text_json.get('analysis', ''),
                                    'plan': text_json.get('plan', ''),
                                    'commands': text_json.get('commands', []),
                                    'task_complete': text_json.get('task_complete', False)
                                })
                            except (json.JSONDecodeError, TypeError):
                                # If still fails, keep the original text
                                cleaned_parts.append({'text': part['text']})
                        else:
                            cleaned_parts.append({'text': part['text']})
                else:
                    cleaned_parts.append(part)
            
            cleaned['assistant'] = {
                'parts': cleaned_parts,
                'role': message.get('role')
            }
            if message.get('taskId'):
                cleaned['assistant']['taskId'] = message['taskId']
            if message.get('referenceTaskIds'):
                cleaned['assistant']['referenceTaskIds'] = message['referenceTaskIds']
        else:
            cleaned['assistant'] = assistant
    
    # Keep command_executions if present, but exclude the commands field
    if 'command_executions' in msg:
        cmd_exec = msg['command_executions'].copy()
        # Remove the commands field as it's redundant with assistant parts
        if 'commands' in cmd_exec:
            del cmd_exec['commands']
        cleaned['command_executions'] = cmd_exec
    
    return cleaned

def extract_conversations(jsonl_path):
    """Extract task_id -> message_history mapping from JSONL file."""
    conversations = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry['task_id']
            message_history = entry['message_history']
            
            # Extract pytest output if it exists
            pytest_output = None
            if 'metadata' in entry:
                eval_details = entry['metadata'].get('evaluation_details', {})
                unit_tests = eval_details.get('unit_tests', {})
                pytest_output = unit_tests.get('raw_output')
            
            # Deduplicate message_history - keep only unique iterations
            # Since each iteration is saved twice, we skip duplicates
            seen_iterations = {}
            deduplicated = []
            for msg in message_history:
                iteration = msg.get('iteration')
                timestamp = msg.get('timestamp')
                key = (iteration, timestamp)
                
                if key not in seen_iterations:
                    seen_iterations[key] = True
                    # Clean the message to remove unnecessary fields
                    deduplicated.append(clean_message(msg))
            
            # Create conversation entry with pytest output if available
            conversation_entry = deduplicated
            if pytest_output:
                # Add pytest output as a special entry at the end
                conversation_entry.append({
                    'pytest_output': pytest_output
                })
            
            conversations[task_id] = conversation_entry
    
    # Save to conversations.json in same directory
    output_path = Path(jsonl_path).parent / "conversations.json"
    with open(output_path, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"Extracted {len(conversations)} conversations")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_conversations.py <jsonl_file_path>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    extract_conversations(jsonl_path)
