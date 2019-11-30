import os

def recursive_chown(path, uid, gid):
    for dirpath, dirnames, filenames in os.walk(path):
        os.chown(dirpath, uid, gid)
        for filename in filenames:
            os.chown(os.path.join(dirpath, filename), uid, gid)

def make_directory(path, uid, gid):
    try:
        os.mkdir(path)
        os.chown(path, uid, gid)
    except OSError:
        print("Creation of directory %s failed" % path)
    else:
        print("Creation of directory %s succeeded" % path)
