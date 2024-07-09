from grobid_client.grobid_client import GrobidClient

if __name__ == "__main__":
    client = GrobidClient(config_path="./config.json")
    client.process("processFulltextDocument", "./paper/test_pdf", output="./paper/test_out/",
                   consolidate_citations=True, tei_coordinates=True, force=True)
