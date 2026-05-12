from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ToolSchemaRef(BaseModel):
    schema_name: str
    fields: List[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    tool_name: str
    description: str
    input_schema: ToolSchemaRef
    output_schema: ToolSchemaRef
    module_name: str


class ToolInvocationRecord(BaseModel):
    invocation_id: str
    tool_name: str
    module_name: str
    input_digest: Dict[str, str] = Field(default_factory=dict)
    output_measurement_ids: List[str] = Field(default_factory=list)
    status: str = "ok"
    error_message: Optional[str] = None
