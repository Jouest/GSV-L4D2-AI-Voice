import os
from send2trash import send2trash
from tqdm import tqdm

if __name__ == '__main__':
    make_sure = input('输入y将audio文件夹下的wav文件移入回收站,输入yes直接永久删除所有wav文件(真的很久):')
    make_sure = make_sure.strip().lower()
    if make_sure in ['yes', 'y']:
        target_dir = '../audio'
        if not os.path.exists(target_dir):
            print(f"错误：目录 {target_dir} 不存在！")
            exit()

        deleted_count = 0
        for root, dirs, files in os.walk(target_dir):
            for file in tqdm(files, desc=f'{root}'):
                if file.lower().endswith('.wav'):
                    try:
                        if make_sure == 'yes':
                            os.remove(os.path.join(root, file))
                        else:
                            send2trash(os.path.join(root, file))
                        deleted_count += 1
                    except Exception as e:
                        print(f"操作失败：{file} - {str(e)}")

        print(f"已处理 {deleted_count} 个文件")
