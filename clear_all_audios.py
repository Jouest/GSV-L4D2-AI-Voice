import os

if __name__ == '__main__':
    make_sure = input('你确定要删除所有音频缓存吗(无法恢复)？如果是，输入y或者yes:')
    if make_sure.lower() in ['yes', 'y']:
        for root, dirs, files in os.walk('audio'):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
