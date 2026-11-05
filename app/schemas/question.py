from pydantic import Field, BaseModel

class QuestionSchema(BaseModel):
    question:str = Field(min_length=5, max_length=1000)
    document_id:str = Field(min_length=1)