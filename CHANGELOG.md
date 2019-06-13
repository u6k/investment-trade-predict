# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
