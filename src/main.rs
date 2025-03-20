mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::{path::PathBuf, process::exit};
use tokenizers::Tokenizer;

// fn main(){
//     let mut a:Tensor<f32> = Tensor::new(vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.], &vec![2,2,3]);
//     a.print();
//     operators::rope(&mut a, 0, 0.);
//     a.print();
// }

fn main() {
    println!("选择你需要的模型");
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
    match buf.trim() {
        "story" => {
            println!("你选择的模型是story");
            story();
        }
        "chat" => {
            println!("你选择的模型是chat");
            chat();
        }
        _ => {println!("your input is unsupported {}", buf.trim());
    exit(0);}
    }
}

fn story() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    // println!("{:?}",model_dir);
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // println!("{:#?}",tokenizer);
    // let input = "Once upon a time";

    loop {
        print!("user input : " );
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        let binding = tokenizer.encode(buf.trim(), true).unwrap();
        // println!("{:#?}",binding);
        let input_ids = binding.get_ids();
        // print!("\n{}", buf.trim());
        let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.);
        println!(
            "llama output : {} {}",
            buf.trim(),
            tokenizer.decode(&output_ids, true).unwrap()
        );
    }
}
fn chat() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut kv_cache = llama.new_cache();
    let mut output_ids;

    // println!("加载成功");
    loop {
        print!("user input : ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        let input = buf.trim();
        if input.eq_ignore_ascii_case("exit") {
            println!("程序退出");
            break;
        }
        let input = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            input
        );

        let binding = tokenizer.encode(input, true).unwrap();
        // println!("{:#?}",binding);

        let input_ids = binding.get_ids();
        // print!("\n{}\n", buf.trim());
        // println!("完成encode ");
        (output_ids, kv_cache) = llama.chat(input_ids, 500, 0.8, 30, 1., kv_cache);
        println!(
            "llama output : {} {}",
            buf.trim(),
            tokenizer.decode(&output_ids, true).unwrap()
        );
    }
}
