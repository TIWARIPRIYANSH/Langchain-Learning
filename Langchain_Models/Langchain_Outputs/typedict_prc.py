from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int


new_person:Person={'name':"Check",'age':31}
print(new_person['age'])
print(new_person)