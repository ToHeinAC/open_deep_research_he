"""
This module provides a workaround for models that don't support the with_structured_output method.
It can be imported and used in any file where with_structured_output is needed.
"""

import asyncio
import json
import re
from typing import List, Type, TypeVar, cast

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

# Define a type variable for the Pydantic model
T = TypeVar('T', bound=BaseModel)

def custom_structured_output(model, pydantic_schema: Type[T]):
    """
    A workaround for models that don't support with_structured_output method.
    Creates a similar functionality by using a PydanticOutputParser.
    
    Args:
        model: The language model
        pydantic_schema: The Pydantic schema for structured output
        
    Returns:
        A runnable that parses output into the specified schema
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_schema)
    
    # Create a template that instructs the model to output in the required format
    schema_str = str(pydantic_schema.schema())
    
    # Get field descriptions for better prompting
    fields_info = []
    for field_name, field in pydantic_schema.__fields__.items():
        # In Pydantic v1, the description is directly on the field object
        description = getattr(field, 'description', None) or f"The {field_name}"
        fields_info.append(f"- {field_name}: {description}")
    
    field_descriptions = "\n".join(fields_info)
    
    # Create a wrapper class with invoke and ainvoke methods
    class StructuredOutputRunnable:
        def __init__(self, model, schema):
            self.model = model
            self.schema = schema
            self.parser = PydanticOutputParser(pydantic_object=schema)
        
        def invoke(self, messages: List[BaseMessage], **kwargs) -> T:
            # Use asyncio to run the async function
            return asyncio.run(self.ainvoke(messages, **kwargs))
        
        async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> T:
            # Add formatting instructions to the last message
            format_instructions = f"\n\nYou must respond with a JSON object that conforms to this schema:\n{schema_str}\n\nField descriptions:\n{field_descriptions}\n\nYour response must be valid JSON without any explanations or conversation outside the JSON."
            
            # Create a copy of messages to avoid modifying the original
            messages_copy = messages.copy()
            
            # Add the formatting instructions to the last message
            if isinstance(messages_copy[-1], HumanMessage):
                new_content = messages_copy[-1].content + format_instructions
                messages_copy[-1] = HumanMessage(content=new_content)
            else:
                messages_copy.append(HumanMessage(content=format_instructions))
            
            # Call the model
            response = await self.model.ainvoke(messages_copy, **kwargs)
            
            # Parse the response
            try:
                # Extract JSON from the response
                content = response.content
                
                # Try to find JSON in the response
                json_match = re.search(r'```json\n(.+?)\n```|\{.+\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                    # Clean up the JSON string
                    json_str = json_str.strip()
                    # Parse as JSON first to validate
                    parsed_json = json.loads(json_str)
                    # Then use the parser to convert to the Pydantic model
                    return cast(T, self.schema.parse_obj(parsed_json))
                else:
                    # If no JSON pattern found, try direct parsing
                    return cast(T, self.schema.parse_obj(json.loads(content)))
            except Exception as e:
                # If parsing fails, try a more aggressive approach
                try:
                    # Look for anything that might be JSON
                    potential_json = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                    if potential_json:
                        json_str = potential_json.group(0)
                        parsed_json = json.loads(json_str)
                        return cast(T, self.schema.parse_obj(parsed_json))
                    else:
                        raise ValueError(f"Could not extract JSON from response: {content}")
                except Exception as inner_e:
                    raise ValueError(f"Failed to parse structured output: {str(e)}\nResponse content: {content}")
        
        # Support method chaining like with_retry
        def with_retry(self, **kwargs):
            # Return self for simple implementation
            return self
    
    return StructuredOutputRunnable(model, pydantic_schema)

def with_structured_output_safe(model, schema_class):
    """
    A safe wrapper for with_structured_output that falls back to the custom implementation
    if the native method is not available.
    
    Args:
        model: The language model
        schema_class: The Pydantic schema class for structured output
        
    Returns:
        A runnable that parses output into the specified schema
    """
    try:
        return model.with_structured_output(schema_class)
    except (NotImplementedError, AttributeError) as e:
        print(f"with_structured_output not available, using custom implementation: {str(e)}")
        return custom_structured_output(model, schema_class)
