"""Small read-only metrics API for the FedMed dashboard prototype."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='FedMed Metrics API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'],
                   allow_credentials=False, allow_methods=['GET'], allow_headers=['*'])

@app.get('/health')
def health(): return {'status':'ok','service':'fedmed-metrics'}

@app.get('/api/metrics')
def metrics():
    # Demo data is explicitly labelled; wire this endpoint to persisted server metrics
    # before presenting any values as experimental outcomes.
    return {'source':'illustrative_demo_data','history':[
        {'round':1,'loss':1.08,'dice':0.31},
        {'round':2,'loss':0.82,'dice':0.43},
        {'round':3,'loss':0.66,'dice':0.52}]}
