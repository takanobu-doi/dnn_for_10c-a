# dnn_for_10c-a
端点の検出をシミュレーションで生成したデータで学習したNNで行う。
検証用にsumation_track.npyやsumation_result.npyというnpyデータをおいておく。

- prediction で出力されるデータの形式
	sca_a_x, sca_a_y, sca_c_x, sca_c_y, end_a_x, end_a_y, end_c_x, end_c_y [pixel]
- resultの形式
	theta [deg], phi [deg], range [mm], e3 [MeV], ex4 [MeV], 
	sca_a_x, sca_a_y, end_a_x, end_a_y, sca_c_x, sca_c_y, end_c_x, end_c_y [pixel]

推論の結果と実験データを比較するときはデータの形式が異なるので注意しましょう。
