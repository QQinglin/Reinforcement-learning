import requests
import os
import gzip
import shutil

def download_mnist():
    # 定义保存目录
    data_dir = 'data'
    raw_dir = os.path.join(data_dir, 'MNIST', 'raw')

    # 创建目录（如果不存在）
    os.makedirs(raw_dir, exist_ok=True)

    # MNIST 数据文件的 URL（使用 PyTorch 的备用地址）
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    urls = [
        f'{base_url}train-images-idx3-ubyte.gz',
        f'{base_url}train-labels-idx1-ubyte.gz',
        f'{base_url}t10k-images-idx3-ubyte.gz',
        f'{base_url}t10k-labels-idx1-ubyte.gz'
    ]

    # 文件名列表
    filenames = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    # 解压后的文件名
    uncompressed_filenames = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'  # 修正为 idx1
    ]

    # 下载并解压每个文件
    for url, filename, uncompressed_filename in zip(urls, filenames, uncompressed_filenames):
        file_path = os.path.join(raw_dir, filename)
        uncompressed_path = os.path.join(raw_dir, uncompressed_filename)

        # 如果文件已存在，跳过下载
        if os.path.exists(uncompressed_path):
            print(f"{uncompressed_filename} 已存在，跳过下载。")
            continue

        print(f"正在下载 {filename}...")
        # 发送 HTTP 请求下载文件
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # 保存压缩文件
            with open(file_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            print(f"{filename} 下载完成。")

            # 解压文件
            print(f"正在解压 {filename}...")
            with gzip.open(file_path, 'rb') as f_in:
                with open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"{uncompressed_filename} 解压完成。")

            # 可选：删除压缩文件（注释掉以保留）
            # os.remove(file_path)
        else:
            print(f"下载 {filename} 失败，状态码: {response.status_code}")

    print("MNIST 数据下载和解压完成！")

if __name__ == '__main__':
    download_mnist()