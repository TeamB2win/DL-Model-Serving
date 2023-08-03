from typing import Dict, Literal
from fastapi import HTTPException
from schema.errors import ErrorResponse


class HTTP_Exception :
    def __init__(self,
                 status_code : Literal, 
                 description : Literal,
                 detail : Literal
                ) -> None :
        self.status_code = status_code
        self.description = description
        self.detail = detail
    
    @property
    def responses(self) -> Dict[str, Dict[str, ErrorResponse]] :
        return {
            self.status_code : {
                'description' : self.description,
                'model' : ErrorResponse
            }
        }
    
    def error_raise(self) :
        raise HTTPException(
            status_code = self.status_code,
            detail = self.detail
        )