# from ushuffle import shuffle, Shuffler
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import random as rn

# filename = './animals_asu_database.fasta'
# filename = []

# for i in range(1, 25):
#     filename.append('./hs' + str(i) + '.fasta')

# print(filename)
# filename = ['./hs1.fasta', './hs2.fasta', './hs3.fasta', './hs4.fasta', './hs5.fasta', './hs6.fasta', './hs7.fasta', './hs8.fasta']
filenameaw = []
filenamear = []
filenamehs = []
filenamehv = []
filenamelr = []
filenamemv = []
filenamepf = []
filenamepv = []
filenameo = []

for i in range(1, 11):
    filenameaw.append('./aw' + str(i) + '.fasta')
    filenamear.append('./ar' + str(i) + '.fasta')
    filenamehs.append('./hs' + str(i) + '.fasta')
    filenamehv.append('./hv' + str(i) + '.fasta')
    filenamelr.append('./lr' + str(i) + '.fasta')
    filenamemv.append('./mv' + str(i) + '.fasta')
    filenamepf.append('./pf' + str(i) + '.fasta')
    filenamepv.append('./pv' + str(i) + '.fasta')
    filenameo.append('./neg' + str(i) + '.fasta')


for j in range (0, 10):
    sequences = []
    sequences = sequences + ([l for l in SeqIO.parse(filenameaw[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamear[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamehs[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamehv[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamelr[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamemv[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamepf[j],'fasta')])
    sequences = sequences + ([l for l in SeqIO.parse(filenamepv[j],'fasta')])
    seqRead = []

    # print(sequences)

    # for m in range (0, len(sequences)):
        #seqRead.append(SeqRecord(Seq(str(sequences[m].seq)), id=sequences[m].id, name=sequences[m].name, description=sequences[m].description))

    SeqIO.write(sequences, filenameo[j], "fasta")
