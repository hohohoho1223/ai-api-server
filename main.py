from typing import Union

from fastapi import FastAPI

import model # model.py를 가져온다

and_model = model.AndModel() # 그안에 있는 AndModel클래스의 인스턴스를 생성한다
or_model = model.OrModel()
xor_model = model.XorModel()

#API 서버 생성
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}
    # return ["Hello": "World"]


@app.get("/items/{item_id}") #endpoint 엔드포인트 부르는 주소 -> 경로 파라메터인데 단순한 값만 넣을떄만 쓰는듯
# /items/{item_id} 경로
# item_id 경로 매개변수(파라미터)
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/left/{left}/right/{right}") #model 실습
def predict(left: int, right: int):
    result = model.predict([left,right])
    return {"result": result}

@app.post("/train") # 생성 기능은 POST로 진행
def train():
    model.train()
    return {"result": "okay!"}

