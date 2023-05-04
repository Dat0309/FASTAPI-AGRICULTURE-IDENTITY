from typing import Optional, List
from base.schema import PaginationInfo
from base.models import BaseModel


class AgricultureRecognitionInfo(BaseModel):
    common_name: Optional[str]

class AgricultureRecognitionInfoListOut(PaginationInfo):
    list: List[AgricultureRecognitionInfo]
