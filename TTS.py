import json
import os
import re
import base64
import io
import tempfile
import speech_recognition as sr
from moviepy.editor import VideoFileClip, concatenate_videoclips
from difflib import SequenceMatcher

class SignLanguageVideoGenerator:
    def __init__(self, signs_dict_path):
        with open(signs_dict_path, 'r', encoding='utf-8') as f:
            self.signs_dict = json.load(f)
        
        self.motion_threshold_percentile = 20
        self.trim_start_buffer = 0.1
        self.trim_end_buffer = 0.1
        
        self.similarity_threshold = 0.75  
        self.fuzzy_threshold = 0.70      
        
        self._build_search_index()
    
    def _build_search_index(self):
        self.normalized_dict = {}
        self.word_variants = {}
        
        for key, value in self.signs_dict.items():
            normalized = self.normalize_text(key)
            self.normalized_dict[normalized] = (key, value)
            
            variants = self._generate_word_variants(key)
            for variant in variants:
                if variant not in self.word_variants:
                    self.word_variants[variant] = []
                self.word_variants[variant].append((key, value))
    
    def _generate_word_variants(self, word):
        variants = set()
        normalized = self.normalize_text(word)
        variants.add(normalized)
        
        if word.startswith('ال'):
            variants.add(word[2:])
            variants.add(self.normalize_text(word[2:]))
        
        if not word.startswith('ال'):
            variants.add('ال' + word)
            variants.add('ال' + normalized)
        
        without_suffix = self.remove_common_suffixes(word)
        if without_suffix != normalized:
            variants.add(without_suffix)
            variants.add('ال' + without_suffix)
        
        return variants
    
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def similarity_ratio(self, s1, s2):
        ratio = SequenceMatcher(None, s1, s2).ratio()
        
        max_len = max(len(s1), len(s2))
        if max_len > 0:
            distance = self.levenshtein_distance(s1, s2)
            lev_ratio = 1 - (distance / max_len)
            ratio = (ratio + lev_ratio) / 2
        
        return ratio
    
    def find_fuzzy_match(self, word, threshold=None):
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        normalized_word = self.normalize_text(word)
        best_match = None
        best_score = 0
        
        for key in self.signs_dict.keys():
            normalized_key = self.normalize_text(key)
            
            score = self.similarity_ratio(normalized_word, normalized_key)
            
            if normalized_word in normalized_key or normalized_key in normalized_word:
                score += 0.1
            
            len_diff = abs(len(normalized_word) - len(normalized_key))
            if len_diff <= 2:
                score += 0.05
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (key, self.signs_dict[key], score)
        
        return best_match
    
    def find_contextual_match(self, word, context_words):
        normalized_word = self.normalize_text(word)
        
        context_patterns = {
            'كيف': {
                'حال': ['كيف حالك', 'حالك'],
                'الحال': ['كيف حالك', 'حالك'],
                'حالك': ['كيف حالك', 'حالك'],
            },
            'شو': {
                'اخبار': ['أخبارك', 'شو أخبارك'],
                'الاخبار': ['أخبارك', 'شو أخبارك'],
                'اخبارك': ['أخبارك', 'شو أخبارك'],
            },
            'وين': {
                'رايح': ['وين رايح', 'رايح'],
                'جاي': ['وين جاي', 'جاي'],
            },
        }
        
        for ctx_word in context_words:
            normalized_ctx = self.normalize_text(ctx_word)
            if normalized_ctx in context_patterns:
                word_map = context_patterns[normalized_ctx]
                if normalized_word in word_map:
                    for mapped_word in word_map[normalized_word]:
                        match = self.find_best_match(mapped_word)
                        if match[0]:
                            return match
        
        direct_match = self.find_best_match(word)
        if direct_match[0]:
            return direct_match
        
        fuzzy_match = self.find_fuzzy_match(word)
        if fuzzy_match:
            return fuzzy_match[1], fuzzy_match[0]
        
        parts = self.split_word_smart(word)
        if parts:
            return None, None  
        
        return None, None
    
    def _check_contextual_phrases(self, word1, word2):
        """فحص الأنماط السياقية وإرجاع قائمة بالعبارات المحتملة"""
        normalized_word1 = self.normalize_text(word1)
        normalized_word2 = self.normalize_text(word2)
        
        contextual_patterns = [
            # كيف + حال/الحال = كيف حالك
            (['كيف'], ['حال', 'الحال', 'حالك'], ['كيف حالك', 'حالك']),
            
            # شو + اخبار/الاخبار = شو أخبارك
            (['شو', 'ش'], ['اخبار', 'الاخبار', 'اخبارك'], ['شو أخبارك', 'أخبارك']),
            
            # وين + رايح/جاي
            (['وين'], ['رايح', 'جاي'], ['وين رايح', 'وين جاي', 'رايح', 'جاي']),
            
            # صباح + الخير
            (['صباح'], ['الخير', 'خير'], ['صباح الخير']),
            
            # مساء + الخير
            (['مساء'], ['الخير', 'خير'], ['مساء الخير']),
            
            # تصبح + على + خير (سيتم معالجتها لاحقاً)
            (['تصبح'], ['علي', 'على'], ['تصبح على خير']),
        ]
        
        phrases_to_try = []
        
        for first_words, second_words, result_phrases in contextual_patterns:
            if normalized_word1 in first_words and normalized_word2 in second_words:
                phrases_to_try.extend(result_phrases)
        
        return phrases_to_try
    
    def normalize_text(self, text):
        arabic_diacritics = re.compile(r'[ّ َ ً ُ ٌ ِ ٍ ْ ـ]')
        text = re.sub(arabic_diacritics, '', text)
        text = re.sub('[إأٱآا]', 'ا', text)
        text = re.sub('ى', 'ي', text)
        text = re.sub('ة', 'ه', text)
        return text
    
    def is_removable_suffix(self, word, suffix):
        """فحص إذا اللاحقة قابلة للإزالة"""
        normalized = self.normalize_text(word)
        
        words_with_ha = [
            'اردنيه', 'جميله', 'كبيره', 'صغيره', 
            'طويله', 'قصيره', 'سريعه', 'بطيئه'
        ]
        
        if suffix == 'ه':
            base = normalized[:-1] if normalized.endswith('ه') else normalized
            
            adjectives = [
                'صعب', 'جميل', 'كبير', 'صغير', 
                'طويل', 'قصير', 'سريع', 'بطيء'
            ]
            if base in adjectives:
                return True
            
            if normalized in words_with_ha or normalized.endswith('يه'):
                return False
            
            return True
        
        if suffix in ['ي', 'ك', 'ه', 'نا', 'كم', 'هم', 'هن', 'ها', 'كن']:
            return True
        
        return True
    
    def remove_common_suffixes(self, word):
        normalized = self.normalize_text(word)
        
        suffixes = [
            'كم',   # كتابكم
            'هم',   # كتابهم
            'هن',   # كتابهن
            'نا',   # كتابنا
            'ها',   # كتابها
            'كن',   # كتابكن
            
            'ي',    # أخوي، أختي، كتابي
            'ك',    # كتابك
            'ه',    # كتابه
            
            'ون',   # معلمون
            'ين',   # معلمين
            'ات',   # معلمات
            
            'ان',   # معلمان
            'تان',  # معلمتان
            'تين',  # معلمتين
        ]
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                if len(normalized) > len(suffix) + 1:
                    if suffix == 'ي':

                        exceptions_ending_with_i = [
                            'ليبي', 'مصري', 'عربي', 'اردني', 'سوري',
                            'ماضي', 'حالي', 'ثاني', 'باقي', 'كافي'
                        ]
                        
                        if normalized in exceptions_ending_with_i:
                            continue
                        
                        if len(normalized) >= 2:
                            char_before_i = normalized[-2]
                            if char_before_i in ['و', 'ت', 'ي']:
                                if self.is_removable_suffix(word, suffix):
                                    return normalized[:-len(suffix)]
                    
                    elif self.is_removable_suffix(word, suffix):
                        return normalized[:-len(suffix)]
        
        return normalized
    
    def split_number(self, num_str):
        try:
            num = int(num_str)
        except:
            return []
        
        if num < 0:
            return []
        
        if str(num) in self.signs_dict:
            return [str(num)]
        
        parts = []
        
        if num >= 1000:
            thousands = (num // 1000) * 1000
            if str(thousands) in self.signs_dict:
                parts.append(str(thousands))
                num = num % 1000
        
        if num >= 100:
            hundreds = (num // 100) * 100
            if str(hundreds) in self.signs_dict:
                parts.append(str(hundreds))
                num = num % 100
        
        if num >= 10:
            tens = (num // 10) * 10
            if str(tens) in self.signs_dict:
                parts.append(str(tens))
                num = num % 10
        
        if num > 0:
            if str(num) in self.signs_dict:
                parts.append(str(num))
            else:
                for digit in str(num):
                    if digit in self.signs_dict:
                        parts.append(digit)
        
        return parts
    
    def split_word_smart(self, word):
        normalized = self.normalize_text(word)
        common_prefixes = ['ال', 'و', 'ف', 'ب', 'ك', 'ل']
        
        for prefix in common_prefixes:
            if normalized.startswith(prefix) and len(normalized) > len(prefix) + 2:
                rest = normalized[len(prefix):]
                if rest in self.signs_dict or self.normalize_text(rest) in [
                    self.normalize_text(k) for k in self.signs_dict.keys()
                ]:
                    return [prefix, rest]
        
        if len(normalized) > 6:
            mid = len(normalized) // 2
            for split_point in range(mid - 1, mid + 2):
                if 2 < split_point < len(normalized) - 2:
                    part1 = normalized[:split_point]
                    part2 = normalized[split_point:]
                    
                    found1 = part1 in self.signs_dict or any(
                        self.normalize_text(k) == part1 for k in self.signs_dict.keys()
                    )
                    found2 = part2 in self.signs_dict or any(
                        self.normalize_text(k) == part2 for k in self.signs_dict.keys()
                    )
                    
                    if found1 and found2:
                        return [part1, part2]
        
        return []
    
    def find_best_match(self, word, _recursion_guard=None):
        if _recursion_guard is None:
            _recursion_guard = set()
        
        if word in _recursion_guard:
            return None, None
        _recursion_guard.add(word)
        
        if word in self.signs_dict:
            return self.signs_dict[word], word
        
        normalized_word = self.normalize_text(word)
        for key in self.signs_dict.keys():
            if self.normalize_text(key) == normalized_word:
                return self.signs_dict[key], key
        
        contextual_mappings = {
            'كيف الحال': ['كيف حالك', 'كيف الحال'],
            'شو الاخبار': ['شو أخبارك', 'شو الأخبار'],
            'وين انت': ['وين أنت', 'وينك'],
            'كيف انت': ['كيف أنت', 'كيفك'],
        }
        
        normalized_word_check = self.normalize_text(word)
        for pattern, alternatives in contextual_mappings.items():
            if self.normalize_text(pattern) == normalized_word_check:
                for alt in alternatives:
                    if alt in self.signs_dict:
                        return self.signs_dict[alt], alt
                    for key in self.signs_dict.keys():
                        if self.normalize_text(key) == self.normalize_text(alt):
                            return self.signs_dict[key], key
        
        word_without_suffix = self.remove_common_suffixes(word)
        if word_without_suffix != normalized_word:
            if word_without_suffix in self.signs_dict:
                return self.signs_dict[word_without_suffix], word_without_suffix
            
            for key in self.signs_dict.keys():
                key_normalized = self.normalize_text(key)
                if word_without_suffix == key_normalized:
                    return self.signs_dict[key], key
                
                key_without_suffix = self.remove_common_suffixes(key)
                if word_without_suffix == key_without_suffix:
                    return self.signs_dict[key], key
        
        if not word.startswith('ال') and 'ال' + word not in _recursion_guard:
            word_with_al = 'ال' + word
            for key in self.signs_dict.keys():
                if self.normalize_text(word_with_al) == self.normalize_text(key):
                    return self.signs_dict[key], key
        
        if word.startswith('ال') and len(word) > 2:
            word_without_al = word[2:]
            if word_without_al not in _recursion_guard:
                for key in self.signs_dict.keys():
                    if self.normalize_text(word_without_al) == self.normalize_text(key):
                        return self.signs_dict[key], key
        
        words_in_input = normalized_word.split()
        if len(words_in_input) > 1:
            for key in self.signs_dict.keys():
                if words_in_input == self.normalize_text(key).split():
                    return self.signs_dict[key], key
        
        return None, None
    
    def can_form_longer_phrase(self, current_index, words, max_check=3):
        """
        تحقق إذا الكلمة الحالية ممكن تكوّن عبارة أطول مع الكلمات اللي بعدها
        """
        current_word = words[current_index]
        
        phrase_starters = {
            'كيف': ['حال', 'الحال', 'حالك', 'انت', 'أنت'],
            'شو': ['اخبار', 'الاخبار', 'اخبارك', 'أخبارك'],
            'صباح': ['خير', 'الخير'],
            'مساء': ['خير', 'الخير'],
            'تصبح': ['على', 'علي'],
            'وين': ['رايح', 'جاي', 'انت', 'أنت'],
            'السلام': ['عليكم'],
        }
        
        normalized_current = self.normalize_text(current_word)
        
        if normalized_current not in phrase_starters:
            return False
        
        for i in range(1, min(max_check + 1, len(words) - current_index)):
            next_word = words[current_index + i]
            normalized_next = self.normalize_text(next_word)
            
            if normalized_next in phrase_starters[normalized_current]:
                phrase = ' '.join(words[current_index:current_index + i + 1])
                
                video_path, matched_key = self.find_best_match(phrase)
                if video_path:
                    return True
        
        return False
    
    def text_to_signs(self, text):
        words = text.strip().split()
        video_paths = []
        missing_words = []
        found_matches = []
        used_indices = set()
        
        i = 0
        while i < len(words):
            matched = False
            best_match = None
            
            for phrase_length in range(min(5, len(words) - i), 0, -1):
                if i in used_indices:
                    break
                
                phrase = ' '.join(words[i:i + phrase_length])
                
                video_path, matched_key = self.find_best_match(phrase)
                
                if video_path and matched_key:
                    normalized_phrase = self.normalize_text(phrase)
                    normalized_matched = self.normalize_text(matched_key)
                    
                    if normalized_phrase == normalized_matched:
                        best_match = {
                            'video_path': video_path,
                            'matched_key': matched_key,
                            'phrase': phrase,
                            'length': phrase_length
                        }
                        break
            
            if best_match:
                if best_match['length'] == 1:
                    can_extend = self.can_form_longer_phrase(i, words)
                    if can_extend:
                        matched = False
                        i += 1
                        continue
                
                indices_to_use = set(range(i, i + best_match['length']))
                if not indices_to_use.intersection(used_indices):
                    video_paths.append(best_match['video_path'])
                    used_indices.update(indices_to_use)
                    
                    if best_match['matched_key'] != best_match['phrase']:
                        found_matches.append({
                            'input': best_match['phrase'],
                            'matched': best_match['matched_key'],
                            'type': 'exact_match'
                        })
                    
                    i += best_match['length']
                    matched = True
            
            if not matched and i not in used_indices:
                current_word = words[i]
                
                context_words = []
                if i > 0:
                    context_words.append(words[i-1])
                if i < len(words) - 1:
                    context_words.append(words[i+1])
                
                video_path, matched_key = self.find_contextual_match(current_word, context_words)
                
                if video_path and matched_key:
                    video_paths.append(video_path)
                    used_indices.add(i)
                    
                    # البحث الضبابي
                    fuzzy_info = ""
                    if matched_key != current_word:
                        fuzzy_match = self.find_fuzzy_match(current_word)
                        if fuzzy_match and fuzzy_match[0] == matched_key:
                            fuzzy_info = f" (تشابه: {fuzzy_match[2]:.0%})"
                        
                        found_matches.append({
                            'input': current_word,
                            'matched': matched_key + fuzzy_info,
                            'type': 'smart_match'
                        })
                    
                    i += 1
                    matched = True
                
                # معالجة الأرقام
                if not matched and current_word.isdigit():
                    number_parts = self.split_number(current_word)
                    if number_parts:
                        for part in number_parts:
                            if part in self.signs_dict:
                                video_paths.append(self.signs_dict[part])
                        found_matches.append({
                            'input': current_word,
                            'matched': ' + '.join(number_parts),
                            'type': 'number_split'
                        })
                        used_indices.add(i)
                        matched = True
                
                # البحث الضبابي كخيار أخير
                if not matched:
                    fuzzy_match = self.find_fuzzy_match(current_word, threshold=0.65)
                    if fuzzy_match:
                        video_paths.append(fuzzy_match[1])
                        used_indices.add(i)
                        found_matches.append({
                            'input': current_word,
                            'matched': f"{fuzzy_match[0]} (تشابه: {fuzzy_match[2]:.0%})",
                            'type': 'fuzzy_match'
                        })
                        matched = True
                
                # تقسيم الكلمة
                if not matched:
                    word_parts = self.split_word_smart(current_word)
                    if word_parts:
                        all_found = True
                        temp_paths = []
                        for part in word_parts:
                            part_path, _ = self.find_best_match(part)
                            if not part_path:
                                # محاولة البحث الضبابي للجزء
                                fuzzy = self.find_fuzzy_match(part, threshold=0.65)
                                if fuzzy:
                                    part_path = fuzzy[1]
                            
                            if part_path:
                                temp_paths.append(part_path)
                            else:
                                all_found = False
                                break
                        
                        if all_found:
                            video_paths.extend(temp_paths)
                            used_indices.add(i)
                            found_matches.append({
                                'input': current_word,
                                'matched': ' + '.join(word_parts),
                                'type': 'word_split'
                            })
                            matched = True
                
                if not matched:
                    missing_words.append(current_word)
                
                i += 1
        
        return video_paths, missing_words, found_matches
    
    def decode_base64_audio(self, base64_string):
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        audio_data = base64.b64decode(base64_string)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    
    def speech_to_text(self, audio_path):
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language='ar-SA')
            return text
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def detect_motion_boundaries(self, video_path):
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return 0.15, -0.15
        
        motion_scores = []
        prev_frame = None
        sample_rate = max(1, int(fps / 10))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    motion_scores.append((frame_idx / fps, motion))
                
                prev_frame = gray
            
            frame_idx += 1
        
        cap.release()
        
        if len(motion_scores) < 3:
            return 0.15, -0.15
        
        threshold = np.percentile([m for _, m in motion_scores], self.motion_threshold_percentile)
        
        start_time = 0
        for time, motion in motion_scores:
            if motion > threshold:
                start_time = max(0, time - self.trim_start_buffer)
                break
        
        end_time = motion_scores[-1][0]
        for time, motion in reversed(motion_scores):
            if motion > threshold:
                end_time = min(motion_scores[-1][0], time + self.trim_end_buffer)
                break
        
        return start_time, end_time
    
    def trim_video_smart(self, video_path):
        try:
            start_time, end_time = self.detect_motion_boundaries(video_path)
            clip = VideoFileClip(video_path)
            
            if end_time < 0:
                end_time = clip.duration + end_time
            
            end_time = min(end_time, clip.duration)
            
            if start_time >= end_time or start_time >= clip.duration:
                return clip
            
            return clip.subclip(start_time, end_time)
        except:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            start = min(0.15, duration * 0.15)
            end = max(duration - 0.15, duration * 0.85)
            return clip.subclip(start, end)
    
    def merge_sign_videos(self, video_paths):
        if not video_paths:
            return None
        
        clips = []
        
        first_clip = VideoFileClip(video_paths[0])
        target_width = first_clip.w
        target_height = first_clip.h
        target_fps = first_clip.fps
        first_clip.close()
        
        for path in video_paths:
            if not os.path.exists(path):
                continue
            
            try:
                clip = self.trim_video_smart(path)
                
                if clip.w != target_width or clip.h != target_height:
                    clip = clip.resize((target_width, target_height))
                
                if clip.fps != target_fps:
                    clip = clip.set_fps(target_fps)
                
                clips.append(clip)
            except Exception as e:
                print(f"Error processing video {path}: {e}")
                continue
        
        if not clips:
            return None
        
        final_clip = concatenate_videoclips(clips, method="compose")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            output_path = temp_file.name
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=target_fps,
            preset='ultrafast',
            bitrate='8000k',
            threads=4,
            logger=None
        )
        
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
    
    def encode_video_to_base64(self, video_path):
        try:
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return video_base64
        except Exception as e:
            print(f"Error encoding video: {e}")
            return None
    
    def process_from_flutter(self, input_data, input_type='text'):
        try:
            recognized_text = None
            temp_audio_path = None
            
            if input_type == 'audio':
                temp_audio_path = self.decode_base64_audio(input_data)
                recognized_text = self.speech_to_text(temp_audio_path)
                
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
                if not recognized_text:
                    return {
                        'success': False,
                        'error': 'فشل في التعرف على الصوت',
                        'recognized_text': None,
                        'video_base64': None,
                        'missing_words': [],
                        'found_matches': []
                    }
                
                text = recognized_text
            else:
                text = input_data
            
            video_paths, missing_words, found_matches = self.text_to_signs(text)
            
            if not video_paths:
                return {
                    'success': False,
                    'error': 'لا توجد إشارات متاحة للكلمات المدخلة',
                    'recognized_text': recognized_text,
                    'video_base64': None,
                    'missing_words': missing_words,
                    'found_matches': found_matches,
                    'total_signs': 0
                }
            
            merged_video_path = self.merge_sign_videos(video_paths)
            
            if not merged_video_path:
                return {
                    'success': False,
                    'error': 'فشل في دمج مقاطع الفيديو',
                    'recognized_text': recognized_text,
                    'video_base64': None,
                    'missing_words': missing_words,
                    'found_matches': found_matches,
                    'total_signs': len(video_paths)
                }
            
            video_base64 = self.encode_video_to_base64(merged_video_path)
            
            if os.path.exists(merged_video_path):
                os.remove(merged_video_path)
            
            return {
                'success': True,
                'recognized_text': recognized_text,
                'video_base64': video_base64,
                'missing_words': missing_words,
                'found_matches': found_matches,
                'total_signs': len(video_paths)
            }
            
        except Exception as e:
            print(f"Error in process_from_flutter: {e}")
            return {
                'success': False,
                'error': str(e),
                'recognized_text': None,
                'video_base64': None,
                'missing_words': [],
                'found_matches': []
            }
