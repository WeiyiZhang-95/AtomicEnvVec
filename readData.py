# Functions to read data from .xyz, and LAMMPS files

import os
import numpy as np
from . import GlobalSetting

class xyz():
    '''
    Store data read from .xyz file
    1. Default direction is current direction
    position info in .atom:
        format: [[x],[y],[z], ...]
    species in .species: (*: bond, angle, dihedral, improper)
        format: ['speciesName1', ...]
    '''
    def __init__(self, name, dir=os.getcwd(), file=''):
        self.name = name
        if file:
            file = file
        else:
            file = str(name) + '.xyz'
        self.file = file
        if dir[-1] != '/':
            self.file = dir + '/' + self.file
        else:
            self.file = dir + self.file
        if len(file) < 5 or file not in os.listdir(dir):
            raise FileNotFoundError("xyz file -- (" + file + ") not EXIST!")

    def readAtom(self, timestep = False):
        '''

        :return:
        '''
        f = open(self.file, 'r')
        data = f.readlines()
        tempE = []
        tempA = []
        if timestep == False:
            while data:
                if data[0].find('Timestep') != -1 or data[0].find('timestep') != -1:
                    data.pop(0)
                    break
                if len(data[0].split()) == 4:
                    break
                data.pop(0)
        else:
            timestep = str(timestep)
            while data:
                if data[0].find('Timestep') != -1 or data[0].find('timestep') != -1:
                    if data[0].split(':')[-1].strip() == timestep:
                        data.pop(0)
                        break
                data.pop(0)
        while data:
            if ''.join(data[0]).find('Timesteps:') != -1 and ''.join(data[0]).find('timesteps:') != -1:
                break
            if data[0] == '\n':
                data.pop(0)
                continue
            if len(data[0].split()) < 4:
                break
            tempE.append(data[0].split()[0])
            tempA.append([float(x) for x in data[0].split()[1:]])
            data.pop(0)
        self.atom = np.array(tempA)
        self.species = np.array(tempE)
        return self.atom, self.species

    def read(self):
        '''
        read data from .xyz file
        :return: coordintes and species array read from file
        '''
        return self.readAtom()

class coe():
    '''
    Store data read from .coeff file
    1. The default is to read all.
    2. Default direction is current direction

    pair info in .pair:
        format: [[species1#, species2#, style, [parameters]], ...]
    * info in .*: (*: bond, angle, dihedral, improper)
        format: [*#, style, [parameters]]
    '''
    readSet = {'readPair': True, 'readBond': True, 'readAngle': True, 'readDihedral': True,
               'readImproper': True}

    def __init__(self, name, dir=os.getcwd(), file='', readPair=True, readBond=True, readAngle=True, readDihedral=True,
                 readImproper=True, readAll=False):
        self.name = name
        if file:
            file = file
        else:
            file = str(name) + '.coeff'
        self.file = file
        if dir[-1] != '/':
            self.file = dir + '/' + self.file
        else:
            self.file = dir + self.file
        for e in self.readSet:
            exec('self.readSet[e] = {}'.format(e))
        if readAll == True:
            for e in self.readSet:
                self.readSet[e] = True
        if len(file) < 7 or file not in os.listdir(dir):
            raise FileNotFoundError("coeff file -- (" + file + ") not EXIST!")

    def readPair(self):
        '''
        read pair parameters from LAMMPS input file
        :return: pair parameters
        '''
        f = open(self.file, 'r')
        data = f.readlines()
        f.close()
        temp = []
        while data:
            if len(data[0]) > 5 and data[0][:5] == 'pair_':
                line = data[0].split('#')[0].split()
                temp.append(np.array([line[1], line[2], line[3], [float(x) for x in line[4:]]], dtype='object'))
            data.pop(0)
        self.pair = np.array(temp, dtype='object')
        data.clear()
        return self.pair

    def readF(self, name):
        '''
        read data from File with given label
        :param name: the label to identify if lines needed being read
        :return: data in lines with given label
        '''
        f = open(self.file, 'r')
        data = f.readlines()
        temp = []
        while data:
            if len(data[0]) > len(name) + 1 and data[0][:len(name) + 1] == name + "_":
                line = data[0].split('#')[0].split()
                if len(line) < 5:
                    print('Warning:' + self.file + ': ' + name + 'info might be wrong')
                temp.append(np.array([line[1], line[2], [float(x) for x in line[3:]]], dtype='object'))
            data.pop(0)
        return np.array(temp, dtype='object')

    def readBond(self):
        '''
        read bond parameters
        :return: bond parameters
        '''
        self.bond = self.readF('bond')
        return self.bond
    def readAngle(self):
        '''
        read angel parameters
        :return: angel parameters
        '''
        self.angle = self.readF('angle')
        return self.angle
    def readDihedral(self):
        '''
        read dihedral parameters
        :return: dihedral parameters
        '''
        self.dihedral = self.readF('dihedral')
        return self.dihedral
    def readImproper(self):
        '''
        read improper parameters
        :return: improper parameters
        '''
        self.improper = self.readF('improper')
        return self.improper

    def read(self):
        '''
        read all needed parameter based on the setting
        :return: all needed parameter
        '''
        for e in self.readSet:
            if self.readSet[e]:
                exec('self.{}()'.format(e))

class lmp():
    '''
    Store data read from .lmpdat file
    1. The default is only reading box, elements and atoms data.
    2. Default direction is current direction

    box info in .box:
        format: [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    element info in .element:
        format: [[ele#, mass of the element], ...]
    atom info in .atom and .atom:
        format: [[atom_x,atom_y,atom_z], ...]  #positions
        format: [ele#, ...]
    bond info in .bond:
        format: [[bond type#, [atom1#, atom2#]], ...]
    angle info in .angle:
        format: [[angle type#, [atom1#, atom2#, atom3#]], ...]
    dihedral/improper info in .dihedral/.improper:
        format: [[dihedral/improper type#, [atom1#, atom2#, atom3#, atom4#]], ...]
    '''
    readSet = {'readEle': True, 'readAtom': True, 'readBox': True,
               'readBond': False, 'readAngle': False, 'readDihedral': False,
               'readImproper': False}

    def __init__(self, name, dir=os.getcwd(), file='', readEle=True, readAtom=True, readBox=True,
                 readBond=False, readAngle=False, readDihedral=False, readImproper=False, readAll=False):
        self.name = name
        if file:
            file = file
        else:
            file = str(name) + '.lmpdat'
        self.file = file
        if dir[-1] != '/':
            self.file = dir + '/' + self.file
        else:
            self.file = dir + self.file
        for e in self.readSet:
            exec('self.readSet[e] = {}'.format(e))
        if readAll == True:
            for e in self.readSet:
                self.readSet[e] = True
        if len(file) < 8 or file not in os.listdir(dir):
            raise FileNotFoundError("lmpdat file -- (" + file + ") not EXIST!")

    def readFile(self, funcName):
        '''
        Goes to the starting point to read data file
        :param funcName:
        :return:
        '''
        f = open(self.file, 'r')
        data = f.readlines()
        f.close()
        while data:
            if data[0] == '\n':
                data.pop(0)
                continue
            else:
                exec('self.{}(data)'.format(funcName))
            if data:
                data.pop(0)
        data.clear()
        return

    def readBox(self, *args):
        '''
        read box parameters
        :param args: shows if being called from the class or straightly
        :return: box parameters
        '''
        box = []
        if len(args) == 0:
            self.readFile('readBox')
            return
        else:
            data = args[0]
        if len(data[0].split()[-1]) < 2 or data[0].split()[-1][-2:] != 'hi':
            return False
        while data[0] == '\n' or data[0].split()[-1][-2:] == 'hi':
            if data[0] != '\n':
                box.append([float(data[0].split()[0]), float(data[0].split()[1])])
            data.pop(0)
            if data:
                continue
            else:
                break
        self.box = np.array(box, dtype='object')
        if len(self.box) != 3:
            print("Warning: lmpdat BOX info might be incorrect.")
        GlobalSetting.__statusOutput__(self.file + ': box data loaded from lmpdat file')
        return self.box

    def readEle(self, *args):
        '''
        read elements parameters
        :param args: shows if being called from the class or straightly
        :return: elements parameters
        '''
        if len(args) == 0:
            self.readFile('readEle')
            return
        else:
            data = args[0]
        temp = []
        element = []
        if data[0].split()[0] != 'Masses' and data[0].split()[0] != 'masses':
            return False
        else:
            data.pop(0)
        while True:
            try:
                if data[0] == '\n':
                    data.pop(0)
                    continue
                temp = data[0].split()
                element.append([temp[0], float(temp[1])])
                data.pop(0)
            except:
                break
        self.element = np.array(element, dtype='object')
        GlobalSetting.__statusOutput__(self.file + ': element masses data loaded from lmpdat file')
        return self.element

    def readAtom(self, *args):
        '''
        read atoms parameters
        :param args: shows if being called from the class or straightly
        :return: atoms parameters (coordinates and species)
        '''
        if len(args) == 0:
            self.readFile('readAtom')
            return
        else:
            data = args[0]
        temp = []
        atom = []
        type = []
        if data[0].split()[0] != 'Atoms' and data[0].split()[0] != 'atoms':
            return False
        else:
            data.pop(0)
        while True:
            try:
                if data[0] == '\n':
                    data.pop(0)
                    continue
                temp = data[0].split()
                type.append(temp[2])
                atom.append([float(x) for x in temp[-3:]])
                data.pop(0)
            except:
                break
        self.atom = np.array(atom)
        self.type = np.array(type)
        GlobalSetting.__statusOutput__(self.file + ': atoms data loaded from lmpdat file')
        return self.atom, self.type

    def readBond(self, *args):
        '''
        read bond parameters
        :param args: shows if being called from the class or straightly
        :return: bonds shows bonded atoms
        '''
        if len(args) == 0:
            self.readFile('readBond')
            return
        else:
            data = args[0]
        temp = []
        bond = []
        if data[0].split()[0] != 'Bonds' and data[0].split()[0] != 'bonds':
            return False
        else:
            data.pop(0)
        while True:
            try:
                if data[0] == '\n':
                    data.pop(0)
                    continue
                temp = data[0].split()
                bond.append([temp[1], np.array(temp[-2:], dtype='object')])
                data.pop(0)
            except:
                break
        self.bond = np.array(bond, dtype='object')
        GlobalSetting.__statusOutput__(self.file + ': bond data loaded from lmpdat file')
        return self.bond

    def readAngle(self, *args):
        '''
        read angle parameters
        :param args: shows if being called from the class or straightly
        :return: angel data
        '''
        if len(args) == 0:
            self.readFile('readAngle')
            return
        else:
            data = args[0]
        temp = []
        angle = []
        if data[0].split()[0] != 'Angles' and data[0].split()[0] != 'angles':
            return False
        else:
            data.pop(0)
        while True:
            try:
                if data[0] == '\n':
                    data.pop(0)
                    continue
                temp = data[0].split()
                angle.append([temp[1], np.array(temp[-3:], dtype='object')])
                data.pop(0)
            except:
                break
        self.angle = np.array(angle, dtype=object)
        GlobalSetting.__statusOutput__(self.file + ': angle data loaded from lmpdat file')
        return self.angle

    def readDihedral(self, *args):
        '''
        read dihedral parameters
        :param args: shows if being called from the class or straightly
        :return: dihedrals parameters
        '''
        if len(args) == 0:
            self.readFile('readDihedral')
            return
        else:
            data = args[0]
        temp = []
        dihedral = []
        if data[0].split()[0] != 'Dihedrals' and data[0].split()[0] != 'dihedrals':
            return False
        else:
            data.pop(0)
        while True:
            try:
                if data[0] == '\n':
                    data.pop(0)
                    continue
                temp = data[0].split()
                dihedral.append([temp[1], np.array(temp[-4:], dtype='object')])
                data.pop(0)
            except:
                break
        self.dihedral = np.array(dihedral, dtype='object')
        GlobalSetting.__statusOutput__(self.file + ': dihedral data loaded from lmpdat file')
        return self.dihedral

    def readImproper(self, *args):
        '''
        read improper parameters
        :param args: shows if being called from the class or straightly
        :return: atoms involved in impropers
        '''
        if len(args) == 0:
            self.readFile('readImproper')
            return
        else:
            data = args[0]
        temp = []
        impropers = []
        if data[0].split()[0] != 'Impropers' and data[0].split()[0] != 'impropers':
            return False
        else:
            data.pop(0)
        while True:
            try:
                if data[0] == '\n':
                    data.pop(0)
                    continue
                temp = data[0].split()
                impropers.append([temp[1], np.array(temp[-4:], dtype='object')])
                data.pop(0)
            except:
                break
        self.impropers = np.array(impropers, dtype='object')
        GlobalSetting.__statusOutput__(self.file + ': impropers data loaded from lmpdat file')
        return self.impropers

    def read(self):
        '''
        read all data/parameters based on the setting
        :return:
        '''
        f = open(self.file, 'r')
        data = f.readlines()
        f.close()
        while data:
            if data[0] == '\n':
                data.pop(0)
                continue
            for e in self.readSet:
                if self.readSet[e]:
                    exec('self.{}(data)'.format(e))
                    continue
            if data:
                data.pop(0)
        data.clear()
        return

class all:
    '''
    read data/parameters from all files
    '''
    readSet = {'readLmp': True,'readCoe': False,'readXyz': True}
    def __init__(self,name,dir = os.getcwd(),lmpFile = '',coeFile = '',xyzFile = '',readLmp= True,readCoe=False,readXyz=True,readAll = False):
        self.size = 0
        self.mass = 0
        self.box = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.atom = {}
        self.coeff = {}
        self.element = {'Type':[],'Mass':[]}
        self.name = name
        if dir[-1] != '/':
            self.dir = dir + '/'
        else:
            self.dir = dir
        if lmpFile:
            self.lmpFile = lmpFile
        else:
            self.lmpFile = str(name)+'.lmpdat'
        if coeFile:
            self.coeFile = coeFile
        else:
            self.coeFile = str(name)+'.coeff'
        if xyzFile:
            self.xyzFile = xyzFile
        else:
            self.xyzFile = str(name)+'.xyz'

        for e in self.readSet:
            exec('self.readSet[e] = {}'.format(e))
        for e in zip(['Lmp','Xyz','Coe'],[self.lmpFile,self.xyzFile,self.coeFile]):
            if e[1] not in os.listdir(dir):
                print('Warning: '+e[0]+'file: '+ e[1] +' not exist! Will not read '+e[1])
                self.readSet['read'+e[0]] = False
        if readAll == True:
            for e in self.readSet:
                self.readSet[e] = True

    def readLmp(self):
        '''
        read data from .lmpdat file
        :return:
        '''
        lmpData = lmp(self.name,dir =self.dir, file = self.lmpFile)
        lmpData.readSet['readBond'] = True
        lmpData.read()
        self.atom['lmpPosition'] = lmpData.atom
        self.atom['TypeNo'] = lmpData.type
        self.box = lmpData.box
        self.element['Mass'] = lmpData.element[:,1]
        temp = []
        for i in range(len(self.atom['TypeNo'])):
            temp.append(lmpData.element[:,1][int(lmpData.type[i])-1])
        self.atom['Mass'] = np.array(temp)
        temp.clear()
        self.bond = lmpData.bond
    def readXyz(self):
        '''
        read from .xyz file
        :return:
        '''
        xyzData = xyz(self.name,dir =self.dir, file = self.xyzFile)
        xyzData.read()
        self.atom['Type'] = xyzData.species
        self.atom['xyzPosition'] = xyzData.atom
    def readCoe(self):
        '''
        read from .coeff file
        :return:
        '''
        coeData = coe(self.name,dir =self.dir, file = self.coeFile, readAll=True)
        coeData.read()
        for e in ('pair','bond','angle','dihedral','improper'):
            exec ("self.coeff['"+e+"'] = coeData."+e)
    def read(self):
        '''
        read all needed files
        :return:
        '''
        for e in self.readSet:
            if self.readSet[e]:
                exec ('self.{}()'.format(e))

def test():
    x = xyz('x', dir = '../example/', file='linker0-stretched.xyz')
    x.readAtom(timestep=5673356)
    print(x.atom)


if __name__ == '__main__':
    test()
