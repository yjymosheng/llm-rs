{
  description = "A big flake for all the coding language";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # language-modules.url = "path:./language_modules";

  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      language_modules_dir = ./language_modules;

      need_language = [
        # "c"
        # "rust"
        # "haskell"
        # "nodejs"
      ];
      # 如果添加新的语言 overlay
      # 在下面添加 xx-overlay.overlays.default
      # 语言 overlay 映射表
      languageOverlays = {
        rust = inputs.rust-overlay.overlays.default;
      };

      # 根据 need_language 生成 overlays
      input_overlays = builtins.map (lang: languageOverlays.${lang}) (
        builtins.filter (lang: builtins.hasAttr lang languageOverlays) need_language
      );
      # 仅仅需要修改语言配置, 如果需要更多 pkgs .
      # 添加 底部 packages
      # 或者 修改对应modules下的nix文件

      foldFns = {
        packages = {
          init = [ ];
          fn = acc: elem: acc ++ elem;
        };
        env = {
          init = { };
          fn = acc: elem: acc // elem;
        };
        path = {
          init = [ ];
          fn = acc: elem: acc ++ elem;
        };
        overlay = {
          init = [ ];
          fn = acc: elem: acc ++ [ elem ];
        };
      };

      system = "x86_64-linux";

      final-overlays = input_overlays ++ self.outputs.lib.collectOption "overlay";

      pkgs = import nixpkgs {
        inherit system;
        overlays = final-overlays;
      };

      # 解析 临时 path 路径
      # 相对路径写法: 相对于 本flake.nix 所在目录开始写
      # 绝对路径写法: 直接写
      shell_path = [
        # "/home/mosheng/.cargo"
      ] ++ self.outputs.lib.collectOption "path";

    in
    {

      lib = {
        # getAttribute = need_languages: builtins.map ( )  need_langages;
        _getAttribute =
          foldFns: get_option: language_list: language_modules_dir:
          let
            import_file = builtins.map (x: "${language_modules_dir}/${x}.nix") language_list;
            import_result = builtins.map (x: x { inherit pkgs; }) (builtins.map import import_file);
            fold_elements = builtins.map (x: x.${get_option}) import_result;
            fold_function = foldFns.${get_option};
          in
          builtins.foldl' fold_function.fn fold_function.init fold_elements;

        collectOption =
          get_option: self.outputs.lib._getAttribute foldFns get_option need_language language_modules_dir;
      };

      devShells."${system}".default = pkgs.mkShell {
        packages = [ ] ++ self.outputs.lib.collectOption "packages";

        env = { } // self.outputs.lib.collectOption "env";

        # 拼接 path 路径
        shellHook = ''
          export PATH="${builtins.concatStringsSep ":" shell_path}:$PATH"
        '';
      };
    };
}
