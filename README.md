# human-counter
画像保存用リポジトリ

画像リポジトリの作成方法
① 現在のプロジェクトとは別の箇所にディレクトリ作成
② 元のプロジェクトの.gitを作成したディレクトリへコピー
  ex. 元のプロジェクトの.gitがある場所で下記コマンド実行 第２引数はコピー先
  cp -R .git /Users/iwama/Desktop/python/画像
③ 作成したディレクトリでブランチ作成＆移動
  git checkout -b 画像 origin/画像
④ remoteからpull 
  git pull origin 画像
⑤画像追加して add して commit して origin 画像 へ push
