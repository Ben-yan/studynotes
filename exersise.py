#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:26:21 2017

@author: dandan
"""

def layer1(a):
    def layer2(b):
        def layer3(c):
            return a+b+c
        return layer3
    return layer2
