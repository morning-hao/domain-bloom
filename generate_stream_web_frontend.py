import sys
import time
import requests

url = "http://localhost:5000/"  # 根据你的实际服务器地址进行替换
data = {
    "question": "怎么提高睡眠质量"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers, stream=True)
count = 0
start = time.time()
if response.status_code == 200:
    for line in response.iter_lines(decode_unicode=True):
        count += 1
        if line:
            line = line.replace('<FN>', '\n')
            print(line, end='')
            sys.stdout.flush()  # 防止缓存住输出显得一次性返回的
            # if ':' or
else:
    print(f"Request failed with status code {response.status_code}")
end = time.time()
print()
print("总时间:", end - start)
print('每次时间:', (end - start)/count)