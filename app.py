import os
from flask import Flask, render_template, request, jsonify, make_response
import google.generativeai as genai
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from urllib.parse import quote
from xhtml2pdf import pisa
from io import BytesIO
import re

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=10)

# --- Configuration ---
# Attempt to get API keys from environment variables,
# which is a best practice for security and Vercel deployment.
api_key = os.environ.get("API_KEY")
responsive_voice_key = os.environ.get("RESPONSIVE_VOICE_KEY")
if not api_key or not responsive_voice_key:
    raise ValueError("API keys not found. Please set 'API_KEY' and 'RESPONSIVE_VOICE_KEY' environment variables.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Helper Functions ---
def generate_image(prompt, width=800, height=600):
    """Generate image using Pollinations AI."""
    try:
        encoded_prompt = quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
        return image_url
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return "https://image.pollinations.ai/prompt/scenic%20view?width=800&height=600&nologo=true"

def generate_all_images_concurrent(story_data):
    """Generate all images concurrently using ThreadPoolExecutor."""
    try:
        # Create tasks for cover and chapter images
        cover_prompt = f"A professional book cover with a title on it, '{story_data['title']}', fantasy art"
        cover_future = executor.submit(generate_image, cover_prompt, 400, 550)

        chapter_futures = [executor.submit(generate_image, chapter['image_prompt']) for chapter in story_data['chapters']]

        # Get results
        cover_image = cover_future.result()
        chapter_images = [future.result() for future in chapter_futures]

        # Assign URLs to story data
        for chapter, image_url in zip(story_data['chapters'], chapter_images):
            chapter['image'] = image_url

        return story_data, cover_image

    except Exception as e:
        print(f"Error in generate_all_images_concurrent: {str(e)}")
        fallback_url = "https://image.pollinations.ai/prompt/scenic%20view?width=800&height=600&nologo=true"
        cover_fallback = "https://image.pollinations.ai/prompt/book%20cover?width=400&height=550&nologo=true"

        for chapter in story_data['chapters']:
            chapter['image'] = fallback_url

        return story_data, cover_fallback


def get_word_definitions(words):
    """Get definitions using Gemini API with improved prompting and error handling."""
    prompt = f"""
    Define the following words clearly and concisely, focusing on their most common meanings in everyday usage.
    Words to define: {', '.join(words)}

    Respond ONLY with a JSON object in this exact format (no other text):
    {{
        "word1": "simple definition here",
        "word2": "simple definition here"
    }}

    Make sure each definition is:
    - Clear and simple
    - 10-15 words maximum
    - Suitable for the general audience
    - Focuses on the most common meaning
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response to ensure it's valid JSON
        # Remove any markdown formatting if present
        if '```json' in response_text:
            response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
        elif '```' in response_text:
            response_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
            
        # Remove any remaining non-JSON text
        response_text = re.search(r'\{.*\}', response_text, re.DOTALL).group(0)
        
        definitions = json.loads(response_text)
        
        # Ensure all requested words have definitions
        final_definitions = {}
        for word in words:
            word_lower = word.lower()
            # Try to find the word in the definitions (case-insensitive)
            matching_key = next((k for k in definitions.keys() if k.lower() == word_lower), None)
            if matching_key and definitions[matching_key].strip():
                final_definitions[word] = definitions[matching_key]
            else:
                # If definition is missing, make another attempt for just this word
                single_word_prompt = f"""
                Define this word clearly and concisely in 10-15 words:
                Word: {word}

                Respond ONLY with the definition (no other text).
                """
                try:
                    retry_response = model.generate_content(single_word_prompt)
                    definition = retry_response.text.strip()
                    if definition:
                        final_definitions[word] = definition
                    else:
                        final_definitions[word] = "No definition available"
                except Exception:
                    final_definitions[word] = "No definition available"
                    
        return final_definitions
        
    except Exception as e:
        print(f"Error getting definitions: {str(e)}")
        return {word: f"Definition not available due to error: {str(e)}" for word in words}


def extract_terminology(text):
    """Extract 4-5 significant terms and get their definitions using Gemini."""
    # Find words that are potentially complex or important
    words = re.findall(r'\b[A-Za-z]{5,}\b', text)
    
    # Remove common words and duplicates
    common_words = {
        'there', 'their', 'would', 'could', 'should', 'about', 'which', 'these', 
        'those', 'were', 'have', 'that', 'what', 'when', 'where', 'while', 'from',
        'been', 'being', 'other', 'another', 'every', 'everything', 'something',
        'anything', 'nothing', 'through', 'although', 'though', 'without', 'within',
        'around', 'before', 'after', 'under', 'over', 'because'
    }
    
    # Filter words
    filtered_words = []
    for word in words:
        word_lower = word.lower()
        if (
            word_lower not in common_words and
            not word.isupper() and  # Skip acronyms
            len(word) >= 6 and  # Focus on longer words
            not any(char.isdigit() for char in word)  # Skip words with numbers
        ):
            filtered_words.append(word)
    
    # Remove duplicates while preserving case
    unique_words = []
    seen_words = set()
    for word in filtered_words:
        if word.lower() not in seen_words:
            unique_words.append(word)
            seen_words.add(word.lower())
    
    # Select the most interesting words (prioritize longer, less common words)
    selected_words = sorted(unique_words, key=lambda x: (len(x), x.lower()), reverse=True)[:5]
    
    if selected_words:
        # Get definitions using improved Gemini function
        definitions = get_word_definitions(selected_words)
        return {k: v for k, v in definitions.items() if v and v != "No definition available"}
    return {}

# --- Story Generation and Editing Functions ---
def generate_story(keywords, genre, num_chapters, tone, style, age_group, include_magic, include_romance, include_conflict):
    """Generate story content using Gemini API with enhanced parameters."""
    prompt = f"""
    Create a captivating story based on the following:
    Keywords: {keywords}
    Genre: {genre}
    Number of chapters: {num_chapters}
    Tone: {tone}
    Writing Style: {style}
    Target Audience Age Group: {age_group}
    Include Magical Elements: {'Yes' if include_magic else 'No'}
    Include Romance: {'Yes' if include_romance else 'No'}
    Include Major Conflicts: {'Yes' if include_conflict else 'No'}

    Format the response exactly like this JSON structure (don't include any other text before or after the JSON):
    {{
        "title": "Story title",
        "author": "AI Author",
        "moral": "The moral of the story",
        "chapters": [
            {{
                "chapter_number": 1,
                "chapter_title": "Chapter title",
                "content": "Chapter content (approximately 2-4 paragraphs, suitable for the target age group and writing style)",
                "image_prompt": "Detailed visual description for a captivating chapter image, reflecting the chapter's mood and style",
                "terminology": {{}}
            }}
        ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text

        # Parse JSON with multiple fallback attempts
        try:
            story_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try finding JSON in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                story_data = json.loads(json_match.group(1))
            else:
                # Try finding content between curly braces
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    story_data = json.loads(json_match.group(0))
                else:
                    story_data = {
                        "title": f"{genre.title()} Story",
                        "author": "AI Author",
                        "moral": "Could not generate a moral for the story",
                        "chapters": [{
                            "chapter_number": 1,
                            "chapter_title": "Chapter 1",
                            "content": "Story generation failed. Please try again.",
                            "image_prompt": "A blank page with some text",
                            "terminology": {}
                        }]
                    }

        # Add terminology to each chapter
        for chapter in story_data['chapters']:
            chapter['terminology'] = extract_terminology(chapter['content'])

        return story_data

    except Exception as e:
        print(f"Error generating story: {str(e)}")
        return {
            "title": f"{genre.title()} Story",
            "author": "AI Author",
            "moral": "Could not generate a moral for the story",
            "chapters": [{
                "chapter_number": 1,
                "chapter_title": "Chapter 1",
                "content": "Story generation failed. Please try again.",
                "image_prompt": "A blank page with some text",
                "terminology": {}
            }]
        }

def regenerate_chapter(story_data, chapter_number, tone, style, include_magic, include_romance, include_conflict):
    """Regenerate a specific chapter."""
    chapter_index = chapter_number - 1
    if not (0 <= chapter_index < len(story_data["chapters"])):
        return "Invalid chapter number", 400
    
    # Previous context for regeneration
    previous_context = ""
    if chapter_index > 0:
        previous_context = story_data['chapters'][chapter_index - 1]['content']

    prompt = f"""
    Regenerate chapter {chapter_number} of the story titled '{story_data['title']}'.
    
    Previous chapter context (if applicable): {previous_context}
    
    Overall Tone: {tone}
    Writing Style: {style}
    Include Magical Elements: {'Yes' if include_magic else 'No'}
    Include Romance: {'Yes' if include_romance else 'No'}
    Include Major Conflicts: {'Yes' if include_conflict else 'No'}

    Format the response exactly like this JSON structure (don't include any other text before or after the JSON):
    {{
        "chapter_number": {chapter_number},
        "chapter_title": "New chapter title",
        "content": "New chapter content (approximately 2-4 paragraphs, consistent with the story's style and tone)",
        "image_prompt": "Detailed visual description for a captivating chapter image, furthering the narrative visually",
        "terminology": {{}}
    }}
    """

    try:
        response = model.generate_content(prompt)
        try:
            new_chapter = json.loads(response.text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                new_chapter = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    new_chapter = json.loads(json_match.group(0))
                else:
                    new_chapter = {
                        "chapter_number": chapter_number,
                        "chapter_title": "Regenerated Chapter",
                        "content": "Failed to regenerate chapter. Please try again.",
                        "image_prompt": "A blank page with some text",
                        "terminology": {}
                    }

        # Add terminology
        new_chapter['terminology'] = extract_terminology(new_chapter['content'])

        # Replace old chapter with new chapter
        story_data['chapters'][chapter_index] = new_chapter

        # Regenerate image for the chapter
        new_chapter['image'] = generate_image(new_chapter['image_prompt'])

        return new_chapter

    except Exception as e:
        print(f"Error regenerating chapter: {str(e)}")
        return {
            "chapter_number": chapter_number,
            "chapter_title": "Regenerated Chapter",
            "content": "Failed to regenerate chapter. Please try again.",
            "image_prompt": "A blank page with some text",
            "terminology": {}
        }

def continue_story(previous_story, num_new_chapters, tone, style, include_magic, include_romance, include_conflict):
    """Generate additional chapters for an existing story with enhanced parameters."""
    prompt = f"""
    Continue this story with {num_new_chapters} more chapters, maintaining the established themes and characters.
    Previous story title: {previous_story['title']}
    Last chapter content: {previous_story['chapters'][-1]['content']}
    Overall Tone: {tone}
    Writing Style: {style}
    Include Magical Elements: {'Yes' if include_magic else 'No'}
    Include Romance: {'Yes' if include_romance else 'No'}
    Include Major Conflicts: {'Yes' if include_conflict else 'No'}

    Format the response exactly like this JSON structure (don't include any other text before or after the JSON):
    {{
        "chapters": [
            {{
                "chapter_number": {len(previous_story['chapters']) + 1},
                "chapter_title": "New chapter title",
                "content": "New chapter content (approximately 2-4 paragraphs, consistent with the story's style and tone)",
                "image_prompt": "Detailed visual description for a captivating chapter image, furthering the narrative visually",
                "terminology": {{}}
            }}
        ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        try:
            new_chapters = json.loads(response.text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                new_chapters = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    new_chapters = json.loads(json_match.group(0))
                else:
                    new_chapters = {
                        "chapters": [{
                            "chapter_number": len(previous_story['chapters']) + 1,
                            "chapter_title": "New Chapter",
                            "content": "Failed to generate new content. Please try again.",
                            "image_prompt": "A blank page with some text",
                            "terminology": {}
                        }]
                    }

        # Add terminology to each new chapter
        for chapter in new_chapters['chapters']:
            chapter['terminology'] = extract_terminology(chapter['content'])

        return new_chapters['chapters']
    except Exception as e:
        print(f"Error continuing story: {str(e)}")
        return [{
            "chapter_number": len(previous_story['chapters']) + 1,
            "chapter_title": "New Chapter",
            "content": "Failed to generate new content. Please try again.",
            "image_prompt": "A blank page with some text",
            "terminology": {}
        }]

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', responsive_voice_key=responsive_voice_key)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        keywords = data.get('keywords')
        genre = data.get('genre')
        story_length = data.get('storyLength')
        tone = data.get('tone')
        style = data.get('style')
        age_group = data.get('ageGroup')
        include_magic = data.get('includeMagic')
        include_romance = data.get('includeRomance')
        include_conflict = data.get('includeConflict')

        # Determine the number of chapters based on story length
        num_chapters = {
            'short': 3,
            'medium': 5,
            'long': 7,
            'grand': 10
        }.get(story_length, 3)

        # Generate story content
        story = generate_story(keywords, genre, num_chapters, tone, style, age_group, include_magic, include_romance, include_conflict)

        # Generate all images concurrently
        story, cover_image = generate_all_images_concurrent(story)

        return jsonify({
            'story': story,
            'cover_image': cover_image
        })
    except Exception as e:
        print(f"Error in generate route: {str(e)}")
        return jsonify({
            'error': 'Failed to generate story',
            'details': str(e)
        }), 500

@app.route('/regenerate', methods=['POST'])
def regenerate():
    try:
        data = request.get_json()
        story_data = data.get('story')
        chapter_number = int(data.get('chapter_number'))
        tone = data.get('tone')
        style = data.get('style')
        include_magic = data.get('includeMagic')
        include_romance = data.get('includeRomance')
        include_conflict = data.get('includeConflict')

        new_chapter = regenerate_chapter(story_data, chapter_number, tone, style, include_magic, include_romance, include_conflict)

        return jsonify({'story': story_data, 'new_chapter': new_chapter})

    except Exception as e:
        print(f"Error in regenerate route: {str(e)}")
        return jsonify({
            'error': 'Failed to regenerate chapter',
            'details': str(e)
        }), 500

@app.route('/continue', methods=['POST'])
def continue_story_route():
    try:
        data = request.get_json()
        previous_story = data.get('previous_story')
        num_new_chapters = int(data.get('num_new_chapters', 3))
        tone = data.get('tone')
        style = data.get('style')
        include_magic = data.get('includeMagic')
        include_romance = data.get('includeRomance')
        include_conflict = data.get('includeConflict')

        # Generate new chapters
        new_chapters = continue_story(previous_story, num_new_chapters, tone, style, include_magic, include_romance, include_conflict)

        # Generate images for new chapters concurrently
        futures = [executor.submit(generate_image, chapter['image_prompt'])
                  for chapter in new_chapters]

        # Wait for all image generations to complete
        for chapter, future in zip(new_chapters, futures):
            chapter['image'] = future.result()

        return jsonify({'new_chapters': new_chapters})
    except Exception as e:
        print(f"Error in continue route: {str(e)}")
        return jsonify({
            'error': 'Failed to continue story',
            'details': str(e)
        }), 500

@app.route('/get_moral', methods=['POST'])
def get_moral():
    try:
        data = request.get_json()
        story_data = data.get('story')
        return jsonify({'moral': story_data['moral']})
    except Exception as e:
        print(f"Error in get_moral route: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve moral',
            'details': str(e)
        }), 500

@app.route('/download', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json()
        story_data = data.get('story')
        cover_image = data.get('cover_image')
        
        # Configure PDF options
        pdf_options = {
            'page-size': 'A4',
            'margin-top': '2.5cm',
            'margin-right': '2cm',
            'margin-bottom': '2.5cm',
            'margin-left': '2cm',
            'encoding': 'UTF-8',
        }
        
        # Generate HTML
        pdf_html = render_template('pdf_template.html', story=story_data, cover_image=cover_image)
        
        # Create PDF
        pdf_buffer = BytesIO()
        pisa.CreatePDF(
            pdf_html, 
            dest=pdf_buffer,
            encoding='utf-8',
        )
        
        pdf_buffer.seek(0)
        
        # Create response
        response = make_response(pdf_buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{story_data["title"].replace(" ", "_")}.pdf"'
        
        return response
        
    except Exception as e:
        print(f"Download PDF Error: {str(e)}")
        return jsonify({
            "error": "Error generating PDF",
            "details": str(e)
        }), 500
# --- Main ---
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
