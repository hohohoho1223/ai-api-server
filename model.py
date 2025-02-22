# import numpy as np
# import pickle
# import torch
# import torch.optim as optim
# import torch.nn as nn


# class AndModel:
#     def __init__(self):
#         # 파라메터
#         self.weights = np.random.rand(2)
#         self.bias = np.random.rand(1)

#     def train(self): # AND
#         learning_rate = 0.1
#         epochs = 20
#         inputs = np.array([[0 ,0], [0, 1], [1, 0], [1, 1]])
#         outputs = np.array([0,0,0,1])        
#         for epoch in range(epochs):
#             for i in range(len(inputs)):
#                 # 총 입력 계산
#                 total_input = np.dot(inputs[i], self.weights) + self.bias
#                 # 예측 출력 계산
#                 prediction = self.step_function(total_input)
#                 # 오차 계산
#                 error = outputs[i] - prediction
#                 print(f'inputs[i] : {inputs[i]}')
#                 print(f'weights : {self.weights}')
#                 print(f'bias before update: {self.bias}')
#                 print(f'prediction: {prediction}')
#                 print(f'error: {error}')
#                 # 가중치와 편향 업데이트
#                 self.weights += learning_rate * error * inputs[i]
#                 self.bias += learning_rate * error
#                 print('====')        

#     def step_function(self, x):
#         return 1 if x >= 0 else 0
    
#     def predict(self, input_data):
#         total_input = np.dot(input_data, self.weights) + self.bias
#         return self.step_function(total_input) 

#     def save_model(self, filename):
#         with open(filename, 'wb') as f:
#             pickle.dump((self.weights,self.bias), f)

#     def load_model(self,filename):
#         with open(filename, 'rb') as f:
#             self.weights, self.bias = pickle.load(f)

# if __name__ =="__main__":
#     and_model = AndModel()
#     and_model.train()
#     and_model.save_model('and_model.pkl') # 모델 저장

# class OrModel:
#     def __init__(self):
#         # 파라메터
#         self.weights = np.random.rand(2)
#         self.bias = np.random.rand(1)

#     def train(self): # OR
#         learning_rate = 0.1
#         epochs = 20
#         inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#         outputs = np.array([0,1,1,1])        
#         for epoch in range(epochs):
#             for i in range(len(inputs)):
#                 # 총 입력 계산
#                 total_input = np.dot(inputs[i], self.weights) + self.bias
#                 # 예측 출력 계산
#                 prediction = self.step_function(total_input)
#                 # 오차 계산
#                 error = outputs[i] - prediction
#                 print(f'inputs[i] : {inputs[i]}')
#                 print(f'weights : {self.weights}')
#                 print(f'bias before update: {self.bias}')
#                 print(f'prediction: {prediction}')
#                 print(f'error: {error}')
#                 # 가중치와 편향 업데이트
#                 self.weights += learning_rate * error * inputs[i]
#                 self.bias += learning_rate * error
#                 print('====')        

#     def step_function(self, x):
#         return 1 if x >= 0 else 0
    
#     def predict(self, input_data):
#         total_input = np.dot(input_data, self.weights) + self.bias
#         return self.step_function(total_input) 

#     def save_model(self, filename):
#         with open(filename, 'wb') as f:
#             pickle.dump((self.weights,self.bias), f)

#     def load_model(self,filename):
#         with open(filename, 'rb') as f:
#             self.weights, self.bias = pickle.load(f)

# if __name__ =="__main__":
#     or_model = OrModel()
#     or_model.train()
#     or_model.save_model('or_model.pkl') # 모델 저장

# class XorModel(nn.Module):
#     def __init__(self):
#         super(XorModel, self).__init__()
#         self.fc1 = nn.Linear(2,2) #입력층 -> 은닉층 
#         self.fc2 = nn.Linear(2,1) #은닉층 -> 출력층
#         self.sigmoid = nn.Sigmoid() # 활성화 함수

#     def forward(self,x):
#         x = self.sigmoid(self.fc1)
#         x = self.sigmoid(self.fc2)
#         return x

# if __name__ == "__main__":
#     # XOR 데이터
#     inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
#     outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
#     # XOR 모델 학습
#     xor_model = XorModel()
#     criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
#     optimizer = optim.SGD(xor_model.parameters(), lr=0.1)
#       # 학습
#     epochs = 10000
#     for epoch in range(epochs):
#         optimizer.zero_grad()  # 경량화
#         outputs_pred = xor_model(xor_inputs)  # 예측
#         loss = criterion(outputs_pred, xor_outputs)  # 손실 계산
#         loss.backward()  # 역전파
#         optimizer.step()  # 파라미터 업데이트

#         if epoch % 1000 == 0:
#             print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

#     # 예측
#     with torch.no_grad():
#         print("XOR Predictions:")
#         for i in range(4):
#             print(f"Input: {xor_inputs[i]}, Prediction: {xor_model(xor_inputs[i]).item():.4f}")
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

    def train(self):  # AND
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i]: {inputs[i]}')
                print(f'weights: {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

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
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):  # OR
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i]: {inputs[i]}')
                print(f'weights: {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

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


if __name__ == "__main__":
    # AND 모델 학습
    and_model = AndModel()
    and_model.train()
    and_model.save_model('and_model.pkl')  # 모델 저장

    # OR 모델 학습
    or_model = OrModel()
    or_model.train()
    or_model.save_model('or_model.pkl')  # 모델 저장

    # XOR 데이터
    xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    xor_outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # XOR 모델 학습
    xor_model = XorModel()
    criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
    optimizer = optim.SGD(xor_model.parameters(), lr=0.1)

    # 학습
    epochs = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()  # 경량화
        outputs_pred = xor_model(xor_inputs)  # 예측
        loss = criterion(outputs_pred, xor_outputs)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이트

        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    # 예측
    with torch.no_grad():
        print("XOR Predictions:")
        for i in range(4):
            print(f"Input: {xor_inputs[i]}, Prediction: {xor_model(xor_inputs[i]).item():.4f}")
