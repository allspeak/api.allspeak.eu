import os
import tables

def appendArray2File(data, path):

    if not os.path.isfile(path):
        col = len(data[0])

        f = tables.open_file(path, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, col))
        array_c.append(data)
        f.close()
    else:
        f = tables.open_file(path, mode='a')
        f.root.data.append(data)
        f.close()

def getCols(f):
    return len(f.root.data[0])


def getRows(f):
    return len(f.root.data)


def openFile(filename):
    return tables.open_file(filename, mode='r')


def getData(f):
    return f.root.data


