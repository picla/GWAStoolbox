#!/bin/python

'''
Script to add phenotype columns to the PLINK generated .fam file.
This to allow for running multivariate gemma.
This script is intended to use in the 'GEMMA pipeline' together with plink processing of genotype and phenotype data.
'''

# Libraries
import argparse
import pandas as pd

# Parameters
parser = argparse.ArgumentParser(description = 'Parse arguments to add multiple phenotypes to PLINK generated .fam file')

parser.add_argument('-F', '--fam', help = 'The PLINK generated .fam file.', required = True)
parser.add_argument('-P', '--pheno', help = 'The fiel containing all required phenotypes', required = True)
parser.add_argument('-C', '--cols', help = 'vector with column numbers to be added as phenotypes to PLINK generated.fam file including the original phenotype already in .fam file.', required = True, nargs = '+')

args = parser.parse_args()

# read all phenotypes
allpheno = pd.read_csv(args.pheno)

# read .fam file
fam = pd.read_csv(args.fam, sep = ' ', header = None)

# merge requested phenotypes to .fam file
famMultivar = fam.merge(allpheno.iloc[:,[0]+[2,3]], how = 'left', left_on = 0, right_on = 'accession')
famMultivar.drop(columns= [5, 'accession'], inplace = True)

# write new .fam file
famMultivar.to_csv(f'{args.fam}.multivar', header=False, index = False, sep = ' ')


