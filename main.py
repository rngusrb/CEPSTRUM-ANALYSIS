from cepstrum import compute_human_cepstrum
from util import compare_all_cepstrum_v2
from scipy.io import wavfile

e_lst = []
i_lst = []
o_lst = []
u_lst = []
# 남자, 여자 각 성별의 (a,e,i,o,u)를 한 모음씩 세트로 비교
for n in ("a", "e", "i", "o", "u"):
    midx = []
    mlist = []
    fidx = []
    flist = []
    for gender in ("male", "female"):
        for i in ("0", "1", "2", "3", "4"):
            name = (
                "./normalized_data/"
                + gender
                + "/"
                + gender
                + "_"
                + i
                + "_"
                + n
                + ".wav"
            )
            fs1, data = wavfile.read(name)
            if gender == "male":
                midx.append(compute_human_cepstrum(data)[0])
                mlist.append(compute_human_cepstrum(data)[1])
            elif gender == "female":
                fidx.append(compute_human_cepstrum(data)[0])
                flist.append(compute_human_cepstrum(data)[1])
            if n == "e":
                e_lst.append(compute_human_cepstrum(data)[1])
            elif n == "i":
                i_lst.append(compute_human_cepstrum(data)[1])
            elif n == "o":
                o_lst.append(compute_human_cepstrum(data)[1])
            elif n == "u":
                u_lst.append(compute_human_cepstrum(data)[1])

    compare_all_cepstrum_v2(
        mlist[0], mlist[1], mlist[2], mlist[3], mlist[4], midx[0], 0, n
    )
    compare_all_cepstrum_v2(
        flist[0], flist[1], flist[2], flist[3], flist[4], fidx[0], 1, n
    )
