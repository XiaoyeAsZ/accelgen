pycmd << "python3 -c \""
        << "import timeloopfe.v4 as tl; "
        << "spec = tl.Specification.from_yaml_files("
        << "'" << topJinja << "', "
        << "jinja_parse_data={'architecture': '" << archTarget << "', "
        << "'problem': '" << problemPathAbs << "'}); "
        << "tl.call_mapper(spec, output_dir='" << outputDir << "', "
        << "dump_intermediate_to='" << outputDir << "')"
        << "\" 2>&1";
