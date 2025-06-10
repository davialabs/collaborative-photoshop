# import asyncio
# import base64

# from agents import Agent, Runner
# from openai_vivalehack.image_modifications import (
#     adjust_luminosity_base64,
#     change_color_scheme,
#     go_to_image_index,
#     next_image,
#     previous_image,
# )
# from openai_vivalehack.model import AgentContext
# from agents.tool import ImageGenerationTool
# from openai.types.responses.tool_param import ImageGeneration


# agent = Agent(
#     name="Image Modifier Agent",
#     tools=[
#         adjust_luminosity_base64,
#         change_color_scheme,
#         go_to_image_index,
#         next_image,
#         previous_image,
#         ImageGenerationTool(
#             tool_config=ImageGeneration(
#                 type="image_generation",
#                 model="gpt-image-1",
#                 size="1024x1024",
#                 quality="high",
#                 background="auto",
#                 output_format="png",
#                 output_compression=100,
#                 partial_images=0,
#                 moderation="auto",
#                 input_image_mask=None,
#             ),
#         ),
#     ],
# )


# async def main():
#     with open(
#         "/Users/rubenillouz/project/openai-vivalehack/image_modifications_test/kermit.jpg",
#         "rb",
#     ) as img_file:
#         image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

#     agent_context = AgentContext(
#         image_b64=image_b64, modified_images_b64=[image_b64], current_image_index=0
#     )

#     result = await Runner.run(
#         starting_agent=agent,
#         context=agent_context,
#         input="Generate a new image of a cat",
#     )
#     print(result.final_output)

#     # Save the modified image as PNG
#     for i, modified_image_b64 in enumerate(
#         result.context_wrapper.context.modified_images_b64
#     ):
#         modified_image_data = base64.b64decode(modified_image_b64)
#         with open(f"modified_image_{i}.png", "wb") as f:
#             f.write(modified_image_data)

#     final_image_b64 = result.context_wrapper.context.modified_images_b64[
#         result.context_wrapper.context.current_image_index
#     ]
#     final_image_data = base64.b64decode(final_image_b64)
#     with open("final_image.png", "wb") as f:
#         f.write(final_image_data)


# if __name__ == "__main__":
#     asyncio.run(main())
