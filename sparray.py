"""
Module for sparse arrays using dictionaries. Inspired in part 
by ndsparse (https://launchpad.net/ndsparse) by Pim Schellart

Jan Erik Solem, Feb 9 2010.
solem@maths.lth.se (bug reports and feedback welcome)
"""

#currently in editing by Sahil Garg (sahilgar@usc.edu) for project specific needs

import numpy as np
from constants import *
from config import *
from config_console_output import *


def process_slice(curr_slice, max_val):
    if curr_slice.start is None:
        start = 0
    if curr_slice.step is None:
        step = 1
    if curr_slice.stop is None:
        stop = max_val
    return start, stop, step


class sparray(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """
    def __init__(self, shape, default=0, dtype=float):
        self.default = default #default value of non-assigned elements, hidden default to default change by Sahil
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.data = {} #hidden data to data change by Sahil

    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        if isinstance(index, int):
            index = tuple([index])
        if isinstance(index, tuple):
            #this is specifically meant for sparse psi (prior on interactions)
            index_list = list(index)
            for i in range(len(index)):
                if (not isinstance(index[i], list)) and (not isinstance(index[i], np.ndarray)):
                    if isinstance(index_list[i], int):
                        index_list[i] = [index_list[i]]
                    elif isinstance(index_list[i], slice):
                        range_sss = process_slice(index_list[i], self.shape[i])
                        index_list[i] = range(range_sss[0], range_sss[1], range_sss[2])
                        curr_slice = None
            index = tuple(index_list)
            index_list = None
            len_index = len(index)
            if len(self.shape) == state_change_dim:
                if len_index > state_change_dim:
                    raise AssertionError
                elif len_index < state_change_dim:
                    index_list = list(index)
                    for i in range(len_index, len(self.shape)):
                        index_list.append(range(self.shape[i]))
                    index = tuple(index_list)
                for i0 in index[0]:
                        for i1 in index[1]:
                                for i2 in index[2]:
                                        for i3 in index[3]:
                                                for i4 in index[4]:
                                                    for i5 in index[5]:
                                                        curr_key = (i0, i1, i2, i3, i4, i5)
                                                        # print 'curr_key:', curr_key
                                                        if type(value) == self.__class__:
                                                            self.data[curr_key] = value[curr_key]
                                                        else:
                                                            self.data[curr_key] = value
            elif len(self.shape) == complex_form_dim:
                if len_index > complex_form_dim:
                    raise AssertionError
                elif len_index < complex_form_dim:
                    index_list = list(index)
                    for i in range(len_index, len(self.shape)):
                        index_list.append(range(self.shape[i]))
                    index = tuple(index_list)
                    index_list = None
                for i0 in index[0]:
                    for i1 in index[1]:
                        for i2 in index[2]:
                            for i3 in index[3]:
                                for i4 in index[4]:
                                    for i5 in index[5]:
                                        for i6 in index[6]:
                                            for i7 in index[7]:
                                                for i8 in index[8]:
                                                    curr_key = (i0, i1, i2, i3, i4, i5, i6, i7, i8)
                                                    # print 'curr_key:', curr_key
                                                    if type(value) == self.__class__:
                                                        self.data[curr_key] = value[curr_key]
                                                    else:
                                                        self.data[curr_key] = value
            else:
                raise NotImplementedError
        elif isinstance(index, list) or type(index) == np.ndarray:
            # print 'index:', index
            for curr_index in index:
                curr_index = tuple(curr_index)
                if type(value) == self.__class__:
                    self.data[curr_index] = value[curr_index]
                else:
                    # print 'curr_index:', curr_index
                    self.data[curr_index] = value
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        is_single = True
        if isinstance(index, int):
            index = tuple([index])
        index = list(index)
        for i in range(len(index)):
            if isinstance(index[i], list) or isinstance(index[i], np.ndarray):
                if len(index[i]) > 1:
                    is_single = False
                elif len(index[i]) == 1:
                    index[i] = index[i][0]
                else:
                    raise AssertionError
        index = tuple(index)
        if is_single:
            """ get value at position given in index, where index is a tuple. """
            return self.data.get(index, self.default)
        else:
            out_shape = []
            index_list = list(index)
            for i in range(len(index)):
                if (not isinstance(index_list[i], list)) and (not isinstance(index_list[i], np.ndarray)):
                    if isinstance(index_list[i], int):
                        index_list[i] = [index_list[i]]
                    elif isinstance(index_list[i], slice):
                        #start, stop, step
                        range_sss = process_slice(index_list[i], self.shape[i])
                        index_list[i] = range(range_sss[0], range_sss[1], range_sss[2])
                        curr_slice = None
                out_shape.append(len(index_list[i]))
            index = tuple(index_list)
            index_list = None
            out_shape = tuple(out_shape)
            len_index = len(index)
            if len(self.shape) == state_change_dim:
                if len_index > state_change_dim:
                    raise AssertionError
                elif len_index < state_change_dim:
                    index_list = list(index)
                    out_shape_list = list(out_shape)
                    for i in range(len_index, len(self.shape)):
                        index_list.append(range(self.shape[i]))
                        out_shape_list.append(self.shape[i])
                    index = tuple(index_list)
                    out_shape = tuple(out_shape_list)
                out = self.__class__(shape=out_shape, dtype=self.dtype, default=self.default)
                for i0 in index[0]:
                    for i1 in index[1]:
                        for i2 in index[2]:
                            for i3 in index[3]:
                                for i4 in index[4]:
                                    for i5 in index[5]:
                                        key = (i0, i1, i2, i3, i4, i5)
                                        if key in self.data.keys():
                                            out.data[key] = self.data[key]
            elif len(self.shape) == complex_form_dim:
                if len_index > complex_form_dim:
                    raise AssertionError
                elif len_index < complex_form_dim:
                    index_list = list(index)
                    out_shape_list = list(out_shape)
                    for i in range(len_index, len(self.shape)):
                        index_list.append(range(self.shape[i]))
                        out_shape_list.append(self.shape[i])
                    index = tuple(index_list)
                    out_shape = tuple(out_shape_list)
                out = self.__class__(shape=out_shape, dtype=self.dtype, default=self.default)
                for i0 in index[0]:
                    for i1 in index[1]:
                        for i2 in index[2]:
                            for i3 in index[3]:
                                for i4 in index[4]:
                                    for i5 in index[5]:
                                        for i6 in index[6]:
                                            for i7 in index[7]:
                                                for i8 in index[8]:
                                                    key = (i0, i1, i2, i3, i4, i5, i6, i7, i8)
                                                    out.data[key] = self.data.get(key, self.default)
            else:
                raise NotImplementedError
            return out

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if self.data.has_key(index):
            del(self.data[index])

    def __add__(self, other):
        if other is None:
            raise AssertionError
        """ Add two arrays. """
        if type(other) == self.__class__:
            if self.shape == other.shape:
                out = self.__class__(self.shape, self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                    out.data[k] = out.data[k] + other.default
                out.default = self.default + other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k,self.default)
                    out.data[k] = old_val + other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))
        elif isinstance(other, (float, int, long)):
            out = self.__class__(self.shape, self.dtype)
            out.default = self.default + other
            for k in self.data.keys():
                out.data[k] = self.data[k] + other
            return out
        else:
            raise AssertionError

    def __sub__(self, other):
        if other is None:
            raise AssertionError
        """ Subtract two arrays. """
        if type(other) == self.__class__:
            if self.shape == other.shape:
                out = self.__class__(self.shape, self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                    out.data[k] = out.data[k] - other.default
                out.default = self.default - other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k,self.default)
                    out.data[k] = old_val - other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))
        elif isinstance(other, (int, long, float)):
            out = self.__class__(self.shape, self.dtype)
            out.default = self.default - other
            for k in self.data.keys():
                out.data[k] = self.data[k] - other
            return out
        else:
            raise AssertionError

    def __pos__(self):
        return self

    def __neg__(self):
        out = self.__class__(self.shape, self.dtype)
        out.data = self.data.copy()
        out.default = -self.default
        for k in self.data.keys():
            out.data[k] = -self.data[k]
        return out

    def __radd__(self, other):
        if other is None:
            raise AssertionError
        out = self.__class__(self.shape, self.dtype)
        out.data = self.data.copy()
        out.default = self.default + other
        for k in self.data.keys():
            out.data[k] = self.data[k] + other
        return out

    def __rmul__(self, other):
        if other is None:
            raise AssertionError
        out = self.__class__(self.shape, self.dtype)
        out.data = self.data.copy()
        out.default = other * self.default
        for k in self.data.keys():
            out.data[k] = other * self.data[k]
        return out

    def __rdiv__(self, other):
        if other is None:
            raise AssertionError
        out = self.__class__(self.shape, self.dtype)
        out.data = self.data.copy()
        out.default = other / self.default
        for k in self.data.keys():
            out.data[k] = other / self.data[k]
        return out

    def __rsub__(self, other):
        if other is None:
            raise AssertionError
        out = self.__class__(self.shape, self.dtype)
        out.data = self.data.copy()
        out.default = -self.default + other
        for k in self.data.keys():
            out.data[k] = -self.data[k] + other
        return out

    def __mul__(self, other):
        if other is None:
            raise AssertionError
        """ Multiply two arrays (element wise). """
        if type(other) == self.__class__:
            if self.shape == other.shape:
                out = self.__class__(self.shape, self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                    out.data[k] = out.data[k] * other.default
                out.default = self.default * other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k,self.default)
                    out.data[k] = old_val * other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))
        elif isinstance(other, (int, long, float)):
            out = self.__class__(self.shape, self.dtype)
            out.default = self.default * other
            for k in self.data.keys():
                out.data[k] = self.data[k] * other
            return out
        else:
            raise AssertionError

    def __div__(self, other):
        if other is None:
            raise AssertionError
        """ Divide two arrays (element wise).
            Type of division is determined by dtype. """
        if type(other) == self.__class__:
            if self.shape == other.shape:
                out = self.__class__(self.shape, self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                    out.data[k] = out.data[k] / other.default
                out.default = self.default / other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k,self.default)
                    out.data[k] = old_val / other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))
        elif isinstance(other, (int, long, float)):
            out = self.__class__(self.shape, self.dtype)
            out.default = self.default / other
            for k in self.data.keys():
                out.data[k] = self.data[k] / other
            return out
        else:
            raise AssertionError

    def __truediv__(self, other):
        if other is None:
            raise AssertionError
        """ Divide two arrays (element wise).
            Type of division is determined by dtype. """
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.data = self.data.copy()
            for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                out.data[k] = out.data[k] / other.default
            out.default = self.default / other.default
            for k in other.data.keys():
                old_val = out.data.setdefault(k,self.default)
                out.data[k] = old_val / other.data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __floordiv__(self, other):
        if other is None:
            raise AssertionError
        """ Floor divide ( // ) two arrays (element wise). """
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.data = self.data.copy()
            for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                out.data[k] = out.data[k] // other.default
            out.default = self.default // other.default
            for k in other.data.keys():
                old_val = out.data.setdefault(k,self.default)
                out.data[k] = old_val // other.data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mod__(self, other):
        if other is None:
            raise AssertionError
        """ mod of two arrays (element wise). """
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.data = self.data.copy()
            for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                out.data[k] = out.data[k] % other.default
            out.default = self.default % other.default
            for k in other.data.keys():
                old_val = out.data.setdefault(k,self.default)
                out.data[k] = old_val % other.data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __pow__(self, other):
        if other is None:
            raise AssertionError
        """ power (**) of two arrays (element wise). """
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.data = self.data.copy()
            for k in set.difference(set(out.data.keys()),set(other.data.keys())):
                out.data[k] = out.data[k] ** other.default
            out.default = self.default ** other.default
            for k in other.data.keys():
                old_val = out.data.setdefault(k,self.default)
                out.data[k] = old_val ** other.data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __iadd__(self, other):
        if other is None:
            raise AssertionError
        if type(other) == self.__class__:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] + other.default
                self.default = self.default + other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k,self.default)
                    self.data[k] = old_val + other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))
        elif isinstance(other, (int, long, float)):
            self.default = self.default + other
            for k in self.data.keys():
                self.data[k] = self.data[k] + other

    def __isub__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] - other.default
            self.default = self.default - other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val - other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imul__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] * other.default
            self.default = self.default * other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val * other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __idiv__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] / other.default
            self.default = self.default / other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val / other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __itruediv__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] / other.default
            self.default = self.default / other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val / other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ifloordiv__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] // other.default
            self.default = self.default // other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val // other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imod__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] % other.default
            self.default = self.default % other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val % other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ipow__(self, other):
        if other is None:
            raise AssertionError
        if self.shape == other.shape:
            for k in set.difference(set(self.data.keys()),set(other.data.keys())):
                self.data[k] = self.data[k] ** other.default
            self.default = self.default ** other.default
            for k in other.data.keys():
                old_val = self.data.setdefault(k,self.default)
                self.data[k] = old_val ** other.data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    # def __str__(self):
    #     return str(self.dense())
    #
    def dense(self):
        """ Convert to dense NumPy array. """
        out = self.default * np.ones(self.shape)
        for ind in self.data:
            out[ind] = self.data[ind]
        return out

    def sum(self):
        """ Sum of elements."""
        s = 0
        s = s + (self.default * np.array(self.shape).prod())
        for ind in self.data.keys():
            # print 'ind:', ind
            s = s + (self[ind] - self.default)
        return s

    def prod(self):
        """ Sum of elements."""
        s = 1
        s = s * (self.default ** np.array(self.shape).prod())
        # print 's:', s
        for ind in self.data.keys():
            # print 'ind:', ind
            # print 'self[ind]:', self[ind]
            s = s * (self[ind]/float(self.default))
            # print 's:', s
        return s


if __name__ == "__main__":
    #create a sparse array
    A = sparray((3, 3))
    print 'shape =', A.shape, 'ndim =', A.ndim
    A[(1, 1)] = 10
    A[2, 2] = 10
    
    #access an element
    print A[2, 2]
    
    print 'remove an element...'
    print A
    del(A[2, 2])
    print A
    
    print 'array with different default value...'
    B = sparray((3, 3),default=3)
    print B

    print 'adding...'
    print A+A
    print A+B
    print B+B
    
    print 'subtracting...'
    print A-A
    print A-B
    print B-B
    
    print 'multiplication...'
    print A*A
    print A*B
    print B*B
    
    print 'division...'
    print A/B
    print B/B
    
    print 'mod...'
    print B%B
    print A%B
    
    print 'power...'
    print A**B
    
    print 'iadd...'
    A += B
    print A
    A += A
    print A
    
    print 'sum of elements...'
    print A.sum()
    
    print 'mix with NumPy arrays...'
    print A.dense() * np.ones((3, 3))
    
    print 'Frobenius norm...'
    print sum( (A.dense().flatten()-B.dense().flatten())**2)
    print ((A-B)*(A-B)).sum()
