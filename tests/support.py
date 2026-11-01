"""历史样本入口只接受测试命令的显式配置，不扫描用户电脑或项目旧目录。"""

import os
from pathlib import Path


def local_data_root() -> Path:
    """返回pytest配置的本机资料根目录；未设置时返回永不随源码分发的缺省路径。"""
    return Path(os.environ.get("DDZ_TEST_DATA_DIR", str(Path(__file__).parent / "_unavailable_local_data")))
