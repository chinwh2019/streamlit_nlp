# Import the unittest module
import unittest
# Import the main function from the app
from app import main

# Define a test class that inherits from unittest.TestCase
class TestApp(unittest.TestCase):

    # Define a test method that checks if the main function runs without errors
    def test_main(self):
        # Use a try-except block to catch any exceptions
        try:
            # Call the main function
            main()
            # If no exception is raised, assert True
            self.assertTrue(True)
        except Exception as e:
            # If an exception is raised, assert False and print the exception
            self.assertFalse(True)
            print(e)

    # Define a test method that checks if the main function displays the title
    def test_title(self):
        # Use the st.get_element method to get the title element
        title = st.get_element("title")
        # Assert that the title element is not None
        self.assertIsNotNone(title)
        # Assert that the title element has the text "Transformer visualizer"
        self.assertEqual(title.text, "Transformer visualizer")

    # Define a test method that checks if the main function displays the text areas
    def test_text_areas(self):
        # Use the st.get_element method to get the text area elements
        text_area_1 = st.get_element("text_area", label="Input text")
        text_area_2 = st.get_element("text_area", label="Input another text")
        # Assert that the text area elements are not None
        self.assertIsNotNone(text_area_1)
        self.assertIsNotNone(text_area_2)
        # Assert that the text area elements have the default values
        self.assertEqual(text_area_1.value, "time flies like an arrow")
        self.assertEqual(text_area_2.value, "time flies like an arrow")

    # Define a test method that checks if the main function displays the sidebar radio button
    def test_sidebar_radio(self):
        # Use the st.get_element method to get the sidebar radio element
        sidebar_radio = st.get_element("sidebar_radio", label="Select type of transformer")
        # Assert that the sidebar radio element is not None
        self.assertIsNotNone(sidebar_radio)
        # Assert that the sidebar radio element has the options
        self.assertEqual(sidebar_radio.options, ('encoder', 'encoder_decoder', 'vizbert'))
        # Assert that the sidebar radio element has the default value
        self.assertEqual(sidebar_radio.value, 'encoder')

    # Define a test method that checks if the main function displays the sidebar selectbox
    def test_sidebar_selectbox(self):
        # Use the st.get_element method to get the sidebar selectbox element
        sidebar_selectbox = st.get_element("sidebar_selectbox", label="Select a transformer model for visualization.")
        # Assert that the sidebar selectbox element is not None
        self.assertIsNotNone(sidebar_selectbox)
        # Assert that the sidebar selectbox element has the options
        self.assertEqual(sidebar_selectbox.options, ("deberta-large", "deberta-large-wwm", "bert-ja", "bert-uncased"))
        # Assert that the sidebar selectbox element has the default value
        self.assertEqual(sidebar_selectbox.value, "deberta-large")

    # Define a test method that checks if the main function displays the expander elements
    def test_expanders(self):
        # Use the st.get_element method to get the expander elements
        head_view_expander = st.get_element("expander", label="Head view")
        model_view_expander = st.get_element("expander", label="Model view")
        # Assert that the expander elements are not None
        self.assertIsNotNone(head_view_expander)
        self.assertIsNotNone(model_view_expander)
        # Assert that the expander elements are collapsed by default
        self.assertFalse(head_view_expander.expanded)
        self.assertFalse(model_view_expander.expanded)

# Run the tests
if __name__ == "__main__":
    unittest.main()