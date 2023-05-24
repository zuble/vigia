from utils import globo , xdv


npz_path = xdv.create_train_valdt_test_from_xdvtest_bg_train_a(chunck_fsize = globo.CFG_SINET['chunck_fsize'] , debug=True)
print("\n\nNPZ_PATH\n",npz_path)
