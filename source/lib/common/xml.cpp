// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/xml.hpp"
#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace common
{
Xml::Xml(std::string file_name, const Xml* obj)
: file_name_(std::move(file_name))
, state_(BODY_STATE)
{
    if(obj != nullptr)
    {
        map_      = obj->map_;
        level_    = obj->level_;
        included_ = true;
    }
}

Xml::~Xml()
{
    for(auto& x : stack_)
    {
        x->nodes.clear();
        x->copy.reset();
    }
    if(!map_) return;
    for(auto& [_, nodes] : *map_)
    {
        (void) _;
        for(auto& node : nodes)
        {
            node->nodes.clear();
            node->copy.reset();
        }
    }
}

std::shared_ptr<Xml>
Xml::Create(const std::string& file_name, const Xml* obj)
{
    auto xml = std::make_shared<Xml>(file_name, obj);
    if(xml != nullptr)
    {
        if(xml->Init() != false)
        {
            const std::size_t pos  = file_name.rfind('/');
            const std::string path = (pos != std::string::npos) ? file_name.substr(0, pos + 1) : "";

            xml->PreProcess();
            nodes_t incl_nodes;
            for(const auto& node : xml->GetNodes("top.include"))
            {
                if(node->opts.find("touch") == node->opts.end())
                {
                    node->opts["touch"] = "";
                    incl_nodes.push_back(node);
                }
            }
            for(const auto& incl : incl_nodes)
            {
                const std::string& incl_name = path + incl->opts["file"];
                auto               ixml      = Create(incl_name, xml.get());
                if(!ixml)
                {
                    xml.reset();
                    break;
                }
            }
            if(xml)
            {
                xml->Process();
            }
        }
    }

    return xml;
}

void
Xml::AddExpr(const std::string& full_tag, const std::string& name, const std::string& expr)
{
    const std::size_t pos       = full_tag.rfind('.');
    const std::size_t pos1      = (pos == std::string::npos) ? 0 : pos + 1;
    const std::string level_tag = full_tag.substr(pos1);
    auto              level     = std::make_shared<level_t>();
    (*map_)[full_tag].push_back(level);
    level->tag          = level_tag;
    level->opts["name"] = name;
    level->opts["expr"] = expr;
}

void
Xml::AddConst(const std::string& full_tag, const std::string& name, const uint64_t& val)
{
    std::ostringstream oss;
    oss << val;
    AddExpr(full_tag, name, oss.str());
}

bool
Xml::print_func::operator()(const std::string& global_tag, const std::shared_ptr<level_t>& node)
{
    std::cout << global_tag << ":\n";
    for(auto& opt : node->opts)
    {
        std::cout << global_tag << "." << opt.first << " = " << opt.second << "\n";
    }
    return true;
}

void
Xml::Print() const
{
    std::cout << "XML file '" << file_name_ << "':\n";
    ForEach(print_func{});
}

bool
Xml::Init()
{
    fd_ = open(file_name_.c_str(), O_RDONLY);
    if(fd_ == -1)
    {
        // perror((std::string("open XML file ") + file_name_).c_str());
        return false;
    }

    if(map_ == nullptr)
    {
        map_ = std::make_unique<map_t>();
        AddLevel("top");
    }

    return true;
}

void
Xml::PreProcess()
{
    uint32_t ind = 0;
    char     buf[kBufSize];
    bool     error = false;

    while(true)
    {
        const uint32_t pos  = lseek(fd_, 0, SEEK_CUR);
        uint32_t       size = read(fd_, buf, kBufSize);
        if(size <= 0) break;
        buf[size - 1] = '\0';

        if(strncmp(buf, "#include \"", 10) == 0)
        {
            for(ind = 0; (ind < size) && (buf[ind] != '\n'); ++ind)
            {}
            if(ind < size)
            {
                buf[ind] = '\0';
                size     = ind;
                lseek(fd_, pos + ind + 1, SEEK_SET);
            }

            for(ind = 10; (ind < size) && (buf[ind] != '"'); ++ind)
            {}
            if(ind == size)
            {
                error = true;
                break;
            }
            buf[ind] = '\0';

            AddLevel("include");
            AddOption("file", &buf[10]);
            UpLevel();
        }
    }

    if(error)
    {
        fprintf(stderr, "XML PreProcess failed, line '%s'\n", buf);
        abort();
    }

    lseek(fd_, 0, SEEK_SET);
}

void
Xml::Process()
{
    token_t remainder;

    while(true)
    {
        token_t token = (!remainder.empty()) ? remainder : NextToken();
        remainder.clear();

        // token_t token1 = token;
        // token1.push_back('\0');
        // std::cout << ">>> " << &token1[0] << std::endl;

        // End of file
        if(token.empty()) break;

        switch(state_)
        {
            case BODY_STATE:
                if(token[0] == '<')
                {
                    bool     node_begin = true;
                    unsigned ind        = 1;
                    if(token[1] == '/')
                    {
                        node_begin = false;
                        ++ind;
                    }

                    unsigned i = ind;
                    while(i < token.size())
                    {
                        if(token[i] == '>') break;
                        ++i;
                    }
                    for(unsigned j = i + 1; j < token.size(); ++j)
                        remainder.push_back(token[j]);

                    if(i == token.size())
                    {
                        if(node_begin)
                            state_ = DECL_STATE;
                        else
                            BadFormat(token);
                        token.push_back('\0');
                    }
                    else
                    {
                        token[i] = '\0';
                    }

                    const char* tag = &token[ind];
                    if(node_begin)
                    {
                        AddLevel(tag);
                    }
                    else
                    {
                        Inherit(GetOption("base"));

                        if(strncmp(CurrentLevel().c_str(), tag, strlen(tag)) != 0)
                        {
                            token.back() = '>';
                            BadFormat(token);
                        }
                        UpLevel();
                    }
                }
                else
                {
                    BadFormat(token);
                }
                break;
            case DECL_STATE:
                if(token[0] == '>')
                {
                    state_ = BODY_STATE;
                    for(unsigned j = 1; j < token.size(); ++j)
                        remainder.push_back(token[j]);
                    continue;
                }
                else
                {
                    token.push_back('\0');
                    unsigned j = 0;
                    for(j = 0; j < token.size(); ++j)
                        if(token[j] == '=') break;
                    if(j == token.size()) BadFormat(token);
                    token[j]                = '\0';
                    const std::string key   = token.data();
                    const std::string value = &token[j + 1];
                    AddOption(key, value);
                }
                break;
            default:
            {
                ROCP_ERROR << "XML parser error: wrong state: " << state_;
                abort();
            }
        }
    }
}

bool
Xml::SpaceCheck() const
{
    bool cond = ((buffer_[index_] == ' ') || (buffer_[index_] == '\t'));
    return cond;
}

bool
Xml::LineEndCheck()
{
    bool found = false;
    if(buffer_[index_] == '\n')
    {
        buffer_[index_] = ' ';
        ++file_line_;
        found    = true;
        comment_ = false;
    }
    else if(comment_ || (buffer_[index_] == '#'))
    {
        found    = true;
        comment_ = true;
    }
    return found;
}

Xml::token_t
Xml::NextToken()
{
    token_t token;
    bool    in_string    = false;
    bool    special_symb = false;

    while(true)
    {
        if(data_size_ == 0)
        {
            data_size_ = read(fd_, buffer_, kBufSize);
            if(data_size_ <= 0) break;
        }

        if(token.empty())
        {
            while((index_ < data_size_) && (SpaceCheck() || LineEndCheck()))
            {
                ++index_;
            }
        }
        while((index_ < data_size_) && (in_string || !(SpaceCheck() || LineEndCheck())))
        {
            const char symb      = buffer_[index_];
            bool       skip_symb = false;

            switch(symb)
            {
                case '\\':
                    if(special_symb)
                    {
                        special_symb = false;
                    }
                    else
                    {
                        special_symb = true;
                        skip_symb    = true;
                    }
                    break;
                case '"':
                    if(special_symb)
                    {
                        special_symb = false;
                    }
                    else
                    {
                        in_string = !in_string;
                        if(!in_string)
                        {
                            buffer_[index_] = ' ';
                            --index_;
                        }
                        skip_symb = true;
                    }
                    break;
            }

            if(!skip_symb) token.push_back(symb);
            ++index_;
        }

        if(index_ == data_size_)
        {
            index_     = 0;
            data_size_ = 0;
        }
        else
        {
            if(special_symb || in_string) BadFormat(token);
            break;
        }
    }

    return token;
}

void
Xml::BadFormat(token_t token)
{
    token.push_back('\0');
    ROCP_ERROR << "Error: " << file_name_ << ", line " << file_line_ << ", bad XML token '"
               << token.data() << "'";
    abort();
}

void
Xml::AddLevel(const std::string& tag)
{
    auto level = std::make_shared<level_t>();
    level->tag = tag;
    if(level_)
    {
        level_->nodes.push_back(level);
        stack_.push_back(level_);
    }
    level_ = level;

    std::string global_tag = GlobalTag(tag);
    (*map_)[global_tag].push_back(level_);
}

void
Xml::UpLevel()
{
    level_ = stack_.back();
    stack_.pop_back();
}

void
Xml::Copy(const std::shared_ptr<level_t>& from, const std::shared_ptr<level_t>& to)
{
    auto level = to;
    if(level == nullptr)
    {
        AddLevel(from->tag);
        level = level_;
    }
    level->copy = from;
    level->opts = from->opts;

    for(const auto& node : from->nodes)
    {
        bool              found      = false;
        const std::string name       = GetOption("name", node);
        const std::string global_tag = GlobalTag(level->tag) + "." + node->tag;
        for(const auto& item : (*map_)[global_tag])
        {
            if((name == GetOption("name", item)) || (node == item->copy))
            {
                found = true;
                break;
            }
        }
        if(found == false) Copy(node, nullptr);
    }

    if(to == nullptr) UpLevel();
}

void
Xml::Inherit(const std::string& tag)
{
    if(!tag.empty())
    {
        const std::string global_tag = GlobalTag(tag);
        auto              it         = map_->find(global_tag);
        if(it == map_->end())
        {
            fprintf(
                stderr, "Node \"%s\": Base not found \"%s\"\n", level_->tag.c_str(), tag.c_str());
            abort();
        }
        for(const auto& node : it->second)
        {
            Copy(node, level_);
        }
    }
}

std::string
Xml::CurrentLevel() const
{
    return level_->tag;
}

std::string
Xml::GlobalTag(const std::string& tag) const
{
    std::string global_tag;
    for(const auto& level : stack_)
    {
        global_tag += level->tag + ".";
    }
    global_tag += tag;
    return global_tag;
}

void
Xml::AddOption(const std::string& key, const std::string& value)
{
    level_->opts[key] = value;
}

std::string
Xml::GetOption(const std::string& key, std::shared_ptr<const level_t> level)
{
    level   = (level != nullptr) ? level : level_;
    auto it = level->opts.find(key);
    return (it != level->opts.end()) ? it->second : "";
}
}  // namespace common
}  // namespace rocprofiler
