"""audio_onset.py KEY...: SuperFlux-style onset strength (numpy only) -> ckpt/onset/<KEY>.npz (t, odf_hi, odf_all).
STFT 1024/hop 110 (5 ms) at 22.05 kHz, log magnitude, max filter over 3 bins on the previous frame (vibrato suppression),
positive flux summed over bins > 2 kHz (hi) and all bins."""
import sys,os,wave,numpy as np
B=os.path.expanduser('~/Desktop/RallyIQ/train_v8'); os.makedirs(B+'/ckpt/onset',exist_ok=True)
for k in sys.argv[1:]:
    w=wave.open(f'{B}/audio/{k}.wav'); sr=w.getframerate(); x=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
    N,H=1024,110; win=np.hanning(N).astype(np.float32); nf=(len(x)-N)//H
    hi_bin=int(2000/sr*N); odf_hi=np.zeros(nf,np.float32); odf_all=np.zeros(nf,np.float32); prev=None
    for s in range(0,nf,4096):
        e=min(nf,s+4096); idx=np.arange(s,e)[:,None]*H+np.arange(N)[None,:]
        S=np.log1p(10*np.abs(np.fft.rfft(x[idx]*win,axis=1)))
        if prev is None: prev=S[:1]
        P=np.vstack([prev,S[:-1]]); P=np.maximum(np.maximum(P,np.roll(P,1,1)),np.roll(P,-1,1))
        D=np.maximum(S-P,0); odf_all[s:e]=D.sum(1); odf_hi[s:e]=D[:,hi_bin:].sum(1); prev=S[-1:]
    t=(np.arange(nf)*H+N/2)/sr
    np.savez_compressed(f'{B}/ckpt/onset/{k}.npz',t=t.astype(np.float32),hi=odf_hi,all=odf_all); print(k,nf,flush=True)
