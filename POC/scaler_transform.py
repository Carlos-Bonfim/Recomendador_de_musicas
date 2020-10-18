import pickle

class ScalerTransform( object ):
    def __init__( self ):
        self.scaled = pickle.load(open("modelos_treinados/scaler.pkl", "rb"))
        
    def data_scaler(self, df):
        df = self.scaled.transform(df)
        
        return df