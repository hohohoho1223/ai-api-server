#main.py
from typing import Union
from fastapi import FastAPI
import model # model.py를 가져온다
import torch

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

##########################
@app.get("/predict/and/left/{left}/right/{right}") #model 실습
def predict(left: int, right: int):
    and_result = and_model.predict([left,right])
    return {"AND" : and_result}

@app.get("/predict/or/left/{left}/right/{right}") #model 실습    
def predict(left: int, right: int):
    or_result = or_model.predict([left,right])
    return {"OR" : or_result}

@app.get("/predict/xor/left/{left}/right/{right}") #model 실습    
def predict(left: int, right: int):
    xor_result = xor_model.predict([left,right])
    return {"XOR" : xor_result}
##########################

@app.post("/train/and") # 생성 기능은 POST로 진행
def train_and():
    and_model.train()
    return {"result": "and_train is okay!"}

@app.post("/train/or")
def train_or():
    or_model.train()
    return {"reult" : "or_train is okay!"}

@app.post("/train/xor")  # XOR 모델 학습 엔드포인트
def train_xor():
    xor_model.train()
    # 모델 저장
    torch.save(xor_model.state_dict(), 'xor_model.pth')  # 학습 후 모델 저장
    return {"result": "XOR model training is okay!!!!"}

@app.post("/train/all")  # 모든 모델 학습
def train_all():
    and_model.train()
    or_model.train()
    xor_model.train_model()

    and_model.save_model('and_model.pkl')
    or_model.save_model('or_model.pkl')
    torch.save(xor_model.state_dict(), 'xor_model.pth')  # XOR 모델 저장

    return {
        "result": {
            "and": "and_train is okay!",
            "or": "or_train is okay!",
            "xor": "xor_train is okay!"
        }
    }