import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

# 文件路径
file_path = '/mnt/bn/navigation-vla-data-1/mobile_manipulation/merge-openpi-comet/checkpoints/hf_datasets_cache/hf_datasets_cache/node0/parquet/default-88fcf92a229add52/0.0.0/9c460aabd2aa27d1496e5e38d2060760561f0ac2cd6a110134eefa5b3f153b8d/parquet-train-00000-of-00671.arrow'

def inspect_arrow_file(path):
    print(f"正在读取文件: {path}\n")
    
    # 尝试以 Stream 模式读取 (Hugging Face datasets 缓存通常是这种格式)
    try:
        # 使用 memory_map 提高大文件读取效率
        with pa.memory_map(path, 'rb') as source:
            try:
                reader = ipc.open_stream(source)
                print(">>> 成功以 IPC Stream 格式打开")
            except pa.ArrowInvalid:
                # 如果不是 Stream，尝试 File 格式
                source.seek(0)
                reader = ipc.open_file(source)
                print(">>> 成功以 IPC File 格式打开")

            # 1. 查看表结构 (Schema)
            print("\n--- 数据表结构 (Schema) ---")
            print(reader.schema)

            # 2. 读取数据 (转换为 Pandas DataFrame 以便查看)
            # 注意：如果文件非常大，建议只读取第一个 batch
            # 这里演示读取前几行
            table = reader.read_all()
            df = table.to_pandas()

            print(f"\n--- 数据统计 ---")
            print(f"总行数: {len(df)}")
            print(f"总列数: {len(df.columns)}")
            print(f"列名: {df.columns.tolist()}")

            print("\n--- 前 3 行数据预览 ---")
            # 设置显示选项以避免折叠
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', 1000)
            print(df.head(3))
            
            # 如果包含图像数据，通常是 bytes 类型，打印出来看不清，可以检查列类型
            print("\n--- 列类型信息 ---")
            print(df.dtypes)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    inspect_arrow_file(file_path)