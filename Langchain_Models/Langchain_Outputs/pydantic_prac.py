from pydantic import BaseModel, Field,EmailStr
from typing import Optional
class User(BaseModel):
    id:int=Field(description="The unique identifier for the user")
    name:Optional[str]=Field(default="Anonymous",description="The name of the user")
    email:EmailStr

new_user=User(id=1,email="abc_3@gmail.com")
new_user2=User(id='3',name="Nitish",email="nitish@gmail.com")
print(new_user)
print(new_user2)
print(new_user.name)