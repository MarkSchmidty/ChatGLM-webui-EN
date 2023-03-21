import os
import gradio as gr
from modules import options
from modules.context import ctx
from modules.model import infer

css = "style.css"
script_path = "scripts"
_gradio_template_response_orig = gr.routes.templates.TemplateResponse

def predict(query, max_length, top_p, temperature):
    ctx.limit_round()
    flag = True
    for _, output in infer(
            query=query,
            history=ctx.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
    ):
        if flag:
            ctx.append(query, output)
            flag = False
        else:
            ctx.update_last(query, output)
        yield ctx.rh, ""
    ctx.refresh_last()
    yield ctx.rh, ""

def clear_history():
    ctx.clear()
    return gr.update(value=[])

def apply_max_round_click(max_round):
    ctx.max_rounds = max_round
    return f"Applied: max round {ctx.max_rounds}"

def create_ui():
    reload_javascript()

    def set_language_strings(lang):
        nonlocal language_strings

        language_strings = {
            "en": {
                "prompt": "Enter your content...",
                "max_length": "Max Length",
                "top_p": "Top P",
                "temperature": "Temperature",
                "max_rounds": "Max Rounds",
                "clear_history": "Clear History",
                "sync_history": "Sync History",
                "save_history": "Save History",
                "load_history": "Load History",
                "save_md": "Save as MarkDown",
                "send": "Send",
            },
            "zh": {
                "prompt": "输入你的内容...",
                "max_length": "最大长度",
                "top_p": "Top P",
                "temperature": "温度",
                "max_rounds": "最大对话轮数",
                "clear_history": "清空对话",
                "sync_history": "同步对话",
                "save_history": "保存对话",
                "load_history": "读取对话",
                "save_md": "保存为 MarkDown",
                "send": "发送",
            },
        }[lang]

    def change_language(lang):
        set_language_strings(lang)
        input_message.placeholder = language_strings["prompt"]
        max_length.label = language_strings["max_length"]
        top_p.label = language_strings["top_p"]
        temperature.label = language_strings["temperature"]
        max_rounds.label = language_strings["max_rounds"]
        clear.label = language_strings["clear_history"]
        sync_his_btn.label = language_strings["sync_history"]
        save_his_btn.label = language_strings["save_history"]
        load_his_btn.label = language_strings["load_history"]
        save_md_btn.label = language_strings["save_md"]
        submit.label = language_strings["send"]

    language_strings = {}
    set_language_strings("en")

    language_dropdown = gr.Dropdown(["English", "Chinese"], label="Language", default="English")

    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
        language_dropdown.change(change_language)

        prompt = language_strings["prompt"]
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""<h2><center>ChatGLM WebUI</center></h2>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            max_length = gr.Slider(minimum=4, maximum=4096, step=4, label=language_strings["max_length"], value=2048)
                            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label=language_strings["top_p"], value=0.7)
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label=language_strings["temperature"], value=0.95)

                        with gr.Row():
                            max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label=language_strings["max_rounds"], value=20)
                            apply_max_rounds = gr.Button("✔", elem_id="del-btn")

                        cmd_output = gr.Textbox(label="Command Output")

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear = gr.Button(language_strings["clear_history"])

                        with gr.Row():
                            sync_his_btn = gr.Button(language_strings["sync_history"])

                        with gr.Row():
                            save_his_btn = gr.Button(language_strings["save_history"])
                            load_his_btn = gr.UploadButton(language_strings["load_history"], file_types=['file'], file_count='single')

                        with gr.Row():
                            save_md_btn = gr.Button(language_strings["save_md"])

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                with gr.Row():
                    input_message = gr.Textbox(placeholder=prompt, show_label=False, lines=2, elem_id="chat-input")
                    clear_input = gr.Button("🗑️", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button(language_strings["send"], elem_id="c_generate")

        submit.click(predict, inputs=[
            input_message,
            max_length,
            top_p,
            temperature
        ], outputs=[
            chatbot,
            input_message
        ])

        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[input_message], outputs=[input_message])

        save_his_btn.click(ctx.save_history, outputs=[cmd_output])
        save_md_btn.click(ctx.save_as_md, outputs=[cmd_output])
        load_his_btn.upload(ctx.load_history, inputs=[
            load_his_btn,
        ], outputs=[
            chatbot
        ])
        sync_his_btn.click(lambda x: ctx.rh, inputs=[input_message], outputs=[chatbot])

        apply_max_rounds.click(apply_max_round_click, inputs=[max_rounds], outputs=[cmd_output])

    with gr.Blocks(css=css, analytics_enabled=False) as settings_interface:
        with gr.Row():
            reload_ui = gr.Button("Reload UI")

        def restart_ui():
            options.need_restart = True

        reload_ui.click(restart_ui)

    interfaces = [
        (chat_interface, "Chat", "chat"),
        (settings_interface, "Settings", "settings")
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="ChatGLM") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

    return demo


def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    javascript = ""
    # with open("script.js", "r", encoding="utf8") as js_file:
    #     javascript = f'<script>{js_file.read()}</script>'

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as js_file:
            javascript += f"\n<script>{js_file.read()}</script>"

    # todo: theme
    # if cmd_opts.theme is not None:
    #     javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    def template_response(*args, **kwargs):
        res = _gradio_template_response_orig(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
