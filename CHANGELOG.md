# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [#7227: バックテストではなく単にテストにリネームする](https://redmine.u6k.me/issues/7227)
- [#7115: この手法を検証する…時系列予測を一次元畳み込みを使って解く](https://redmine.u6k.me/issues/7115)
- [#7234: ブレーク法のルールを改善する](https://redmine.u6k.me/issues/7234)
- [#7238: 処理全体の流れを見直す](https://redmine.u6k.me/issues/7238)
- [#7245: csvも圧縮して保存する](https://redmine.u6k.me/issues/7245)
- [#7225: 全銘柄バックテストで、単利運用(費用考慮なし)・単利運用(手数料・税金考慮あり)・複利運用の資産推移を算出する](https://redmine.u6k.me/issues/7225)
- [#7181: 指数平滑移動平均を指標に追加する](https://redmine.u6k.me/issues/7181)
- [#7174: ボリンジャーバンドを指標に追加する](https://redmine.u6k.me/issues/7174)
- [#7177: MACDによるシミュレーションを行う](https://redmine.u6k.me/issues/7177)
- [#7229: 個別銘柄テストで、利益率(通常、必勝)を算出する](https://redmine.u6k.me/issues/7229)
- [#7259: 日経平均などのデータでまずは試す](https://redmine.u6k.me/issues/7259)
- [#7248: 移動平均交差法とデイトレードを組み合わせてシミュレーションしてみる](https://redmine.u6k.me/issues/7248)
- [#7220: データを階差系列に変換して学習する](https://redmine.u6k.me/issues/7220)
- [#7266: 重複データを除去できていない](https://redmine.u6k.me/issues/7266)

## [0.12.0] - 2019-07-20

### Added

- [#7203: バックテスト後に各種指標をレポートする](https://redmine.u6k.me/issues/7203)
- [#7209: 始値で購入、翌日始値で売却する売買アルゴリズムを構築してみる](https://redmine.u6k.me/issues/7209)
- [#7210: 全銘柄を対象としたバックテストを行う](https://redmine.u6k.me/issues/7210)
- [#7212: レポートにリスク(標準偏差)を追加する](https://redmine.u6k.me/issues/7212)
- [#7198: 移動平均交差法で売買シミュレーションを行い、勝つ買いシグナルを学習してみる](https://redmine.u6k.me/issues/7198)
- [#7217: レポートにシャープ・レシオを追加する](https://redmine.u6k.me/issues/7217)
- [#7132: 日経平均やTOPIXなどの株価指数で学習したモデルでバックテストを行ってみる](https://redmine.u6k.me/issues/7132)
- [#7223: 全バックテストで、buyやfeeもprofitを算出する](https://redmine.u6k.me/issues/7223)

## [0.11.0] - 2019-07-15

### Added

- [#7147: この手法を試す…新高値更新のアルゴリズム ](https://redmine.u6k.me/issues/7147)
- [#7202: デイトレードの単一銘柄バックテストを実装する](https://redmine.u6k.me/issues/7202)
- [#7200: ローカル・ストレージではなくS3を使用するようにする](https://redmine.u6k.me/issues/7200)
- [#7204: シミュレーショントレードとバックテストのファイルを統合する](https://redmine.u6k.me/issues/7204)

## [0.10.0] - 2019-07-10

### Added

- [#7157: モメンタムで売買シミュレーションを行う](https://redmine.u6k.me/issues/7157)
- [#7158: ROC(変化率)で売買シミュレーションを行う](https://redmine.u6k.me/issues/7158)
- [#7159: RSI(相対力指数)で売買シミュレーションを行う](https://redmine.u6k.me/issues/7159)
- [#7160: ストキャスティクスで売買シミュレーションを行う](https://redmine.u6k.me/issues/7160)
- [#7183: predictの入力データ、ラベルデータが間違えている](https://redmine.u6k.me/issues/7183)
- [#7162: 入力データのサイズ・スケールを揃える](https://redmine.u6k.me/issues/7162)
- [#7127: 性能評価が高いモデルとデータを使用してバックテストを行う](https://redmine.u6k.me/issues/7127)
- [#7186: 翌日の騰落を予測・売買シミュレーションを行う](https://redmine.u6k.me/issues/7186)
- [#7193: エントリーポイントをスクリプトごとに作成する](https://redmine.u6k.me/issues/7193)
- [#7191: 最初の前処理を、各種整形を行い指標を算出することとする](https://redmine.u6k.me/issues/7191)
- [#7196: ログ出力を適切に行う](https://redmine.u6k.me/issues/7196)
- [#7192: 前処理、学習処理などを並列実行可能にする](https://redmine.u6k.me/issues/7192)

## [0.9.0] - 2019-06-26

### Added

- [#7137: 終値だけで取引したらどうなるか？](https://redmine.u6k.me/issues/7137)
- [#7155: 移動平均を利用した売買アルゴリズムを構築する](https://redmine.u6k.me/issues/7155)

## [0.8.0]

### Added

- [#7153: ごく単純な順張り戦略でバックテストする](https://redmine.u6k.me/issues/7153)

## [0.7.0] - 2019-06-13

### Added

- [#7152: 全銘柄を対象に、最高効率でデイトレードしたらどうなるかバックテストする](https://redmine.u6k.me/issues/7152)

## [0.6.0] - 2019-06-11

### Changed

- [#7133: 銘柄リストに、2018年の開始ID、終了ID、2019年の開始IDが欲しい](https://redmine.u6k.me/issues/7133)
- [#7131: 銘柄選出で直近のOHLCも表示してほしい](https://redmine.u6k.me/issues/7131)

## [0.5.0] - 2019-06-07

### Added

- [#7102: 始値・終値の騰落率を入力データとしたらどうなるか？](https://redmine.u6k.me/issues/7102)

## [0.4.0] - 2019-06-04

### Added

- [#7122: ml-sandboxである程度組み立てたスクリプトをpyに移植する](https://redmine.u6k.me/issues/7122)

## [0.3.0] - 2019-05-31

### Added

- [#7079: 株価予測をランダム・フォレストで試みる](https://redmine.u6k.me/issues/7079)

## [0.2.0] - 2019-05-30

### Added

- [#7062: 売買システムについて、全上場銘柄を対象としたバックテストを行えるようにする](https://redmine.u6k.me/issues/7062)
    - 注目するべき銘柄を抽出する機能を実装しました
    - 複数銘柄のモデルを構築してスコアを算出する機能を実装しました

## [0.1.0] - 2019-04-24

### Fixed

- [#6901: クローラー機能、判断機能のリポジトリを分ける](https://redmine.u6k.me/issues/6901)
    - `investment-machine`から`investment-stocks`に変更して、`predict-prices`から`predict-trend`に変更しました

## [0.0.1] - 2019-04-12

### Added

- [#6954: 空のpipプロジェクトを作成する](https://redmine.u6k.me/issues/6954)
