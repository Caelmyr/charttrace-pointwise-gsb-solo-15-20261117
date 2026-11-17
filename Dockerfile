FROM solo12-scanweave:1.0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CHARTTRACE_DATA_DIR=/data PORT=18918
EXPOSE 18918
CMD ["python3","app.py"]
