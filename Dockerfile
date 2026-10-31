# 构建 Python 3.9 正式 Flask 业务后端镜像，安装依赖后通过 start.sh 启动。
FROM python:3.9

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY . /app

ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["/bin/sh", "start.sh"]
