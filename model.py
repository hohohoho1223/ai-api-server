# model.py
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn as nn


class AndModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])
        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(inputs[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.bias), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.weights, self.bias = pickle.load(f)


class OrModel:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 1])
        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(inputs[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction
                
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.bias), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.weights, self.bias = pickle.load(f)

class XorModel(nn.Module):
    def __init__(self):
        super(XorModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 입력층 -> 은닉층 
        self.fc2 = nn.Linear(2, 1)  # 은닉층 -> 출력층
        self.sigmoid = nn.Sigmoid()  # 활성화 함수

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def train(self, epochs=10000):
        xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        xor_outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        for epoch in range(epochs):
            optimizer.zero_grad()  # 경량화
            outputs_pred = self(xor_inputs)  # 예측
            loss = criterion(outputs_pred, xor_outputs)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 파라미터 업데이트

            if epoch % 1000 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    # 예측
    def predict(self, input_data):  # 예측 메소드 추가
        with torch.no_grad(): #PyTorch의 자동 미분 기능이 비활성화 : 예측할 때는 그래디언트 계산이 필요하지 않기 때문
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            prediction = self(input_tensor).item()  # 예측값 반환(item()함수 : tensor값을 파이썬의 float형식으로 반환
            return 1 if prediction >= 0.5 else 0
        

if __name__ == "__main__": #해당 모듈에서 직접 실행될때를 명시함..!
    # AND 모델 학습
    and_model = AndModel()
    and_model.train()
    and_model.save_model('and_model.pkl')

    # OR 모델 학습
    or_model = OrModel()
    or_model.train()
    or_model.save_model('or_model.pkl')

    # XOR 모델 학습
    xor_model = XorModel()
    xor_model.train()
    torch.save(xor_model.state_dict(), 'xor_model.pth')  # 학습 후 모델 저장
    

