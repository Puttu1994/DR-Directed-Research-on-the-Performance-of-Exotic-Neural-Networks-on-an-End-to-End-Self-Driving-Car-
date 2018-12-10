

class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

MODE = "CNN"
if MODE=="CNN":
	config = dict2(**{
        "dataset":      "comma",
        "imgRow":   	160,
        "imgCol":   	320,
        "imgCh":    	3,
        "resizeFactor": 2,
        "epoch": 	200,
        "epochsize":	30050,#60100,
        "lr": 			1e-3,
        "save_steps": 	10000,
        "val_steps": 	200,
        "model_path": 	"D:\DR\Dataset\Results\Train",
        "pretrained_model_path": None,
        "test_replay":  None,
        "skipvalidate": False,
        "use_curvature":True,
        "UseFeat":      False,
        "CausalityTest":False,
        "Percentage":   0,
        "dim_ctx": 		64, 		# for attn model
        "dim_hidden": 	512, 		# for attn model
        "batch_size": 	80, 		# for attn model
        "timelen": 		1, 			# for attn model
        "ctx_shape":	[200,64], 	# for attn model
        "use_smoothing": "Exp", # smoothing method
        "alpha": 		0.05})
elif MODE=="VA-JOHN":
	config = dict2(**{
        "dataset":      "comma",
        "imgRow":   	160,
        "imgCol":   	320,
        "imgCh":    	3,
        "resizeFactor": 2,
        "epoch": 		256,#128,
        "epochsize":	40100,
        "lr": 			1e-3,
        "save_steps": 	5000,
        "val_steps": 	100,
        "model_path": 	'D:\DR\Dataset\Results\VA_Train',#"/data/TrainingFiles/{}/model/".format("comma"),
        "pretrained_model_path": None,
        "test_replay": 	None,
        "skipvalidate": False,
        "use_curvature":True,
        "UseFeat": 		True,
        "CausalityTest":False,
        "Percentage":   0,
        "dim_ctx": 		64,
        "dim_hidden": 	512,
        "batch_size": 	80,
        "timelen": 		20,
        "ctx_shape":	[200,64],
        "use_smoothing": "Exp",
        "alpha": 		0.05})
else:
	raise NotImplementedError
