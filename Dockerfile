FROM python:3.11-alpine

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias, fixando a versão do LLVM para evitar conflitos
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    gfortran \
    llvm15-dev \
    llvm15-libs \
    clang \
    llvm15

# Define a variável de ambiente para apontar para o llvm-config
ENV LLVM_CONFIG=/usr/lib/llvm15/bin/llvm-config

# Instala numpy primeiro para evitar problemas de dependência com numba e llvmlite
RUN pip install --no-cache-dir numpy

# Instala numba e llvmlite, agora que o llvm-config está disponível
RUN pip install --no-cache-dir numba llvmlite

# Copia o arquivo de requisitos
COPY requirements.txt /app/

# Instala as dependências do requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos da aplicação
COPY app.py /app/
COPY Tensor.py /app/

# Exponha a porta que o aplicativo vai rodar
EXPOSE 8081

# Comando para rodar a aplicação usando waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8081", "app:app"]
